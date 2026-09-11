"""Work package Q: recovery-policy plumbing with fakes, no ROS launch, no simulator."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest
import yaml

from src.experiments.campaigns import expand_balanced_pilot, targeted_execution_order
from src.experiments.recovery_campaigns import (
    PILOT_POLICY_SET, expand_recovery_policies, validate_recovery_pilot,
)
from src.recovery import GuardConfig, RecoveryRequest, RobotState, decide_recovery
from src.recovery.costs import load_cost_weights
from src.recovery.guards import eligible_actions
from src.recovery.plumbing import (
    PILOT_POLICY_IDS, POLICY_IDS, RECOVERY_LAUNCH_ARGUMENTS, forced_action,
    recovery_cli_arguments, recovery_launch_arguments, recovery_system_block,
    resumed_mission_terminal, summarise_recovery_sidecars, validate_evidence_override,
    validate_policy_id,
)
from src.recovery.selector_training import train_selector
from scripts.build_recovery_cost_table import cost_rows
from scripts.build_recovery_outcome_table import build_outcome_rows, rows_to_csv_bytes
from scripts.build_recovery_pilot_manifest import build_manifest, validation_map_routes


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ros_ws/src/recovery_manager"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))
from recovery_manager.live_execution import (  # noqa: E402
    ACTION_SEQUENCES, GateDecision, LiveExecutor, SAFE_STOP_PREFIX,
)


def state(**changes) -> RobotState:
    values = dict(
        stopped=True, stop_allowed=True, localisation_poor=False, planning_stale_or_blocked=False,
        rear_clearance_m=1.0, rotation_clearance_m=1.0, immediate_collision_risk=False,
        obstruction_may_be_transient=False, relocalisation_available=False,
        repeated_recovery_count=0,
    )
    values.update(changes)
    return RobotState(**values)


# ----------------------------------------------------------------------------- policy ids


def test_policy_ids_cover_paired_and_pilot_and_reject_unknown():
    assert POLICY_IDS[:4] == ("R0", "R1", "R2", "R3")
    assert PILOT_POLICY_IDS == tuple(f"RP_{a}" for a in (
        "controlled_stop", "relocalise", "replan_clear_costmaps", "backup", "spin_active_rescan", "wait",
    ))
    assert forced_action("RP_backup") == "backup" and forced_action("R2") is None
    with pytest.raises(ValueError, match="unknown recovery policy"):
        validate_policy_id("R9")
    with pytest.raises(ValueError):
        validate_policy_id("RP_request_assistance")


def test_r0_default_adds_no_arguments_anywhere():
    assert recovery_cli_arguments({"episode_key": "x", "seed": 1}) == []
    pilot = {"episode_key": "x", "seed": 1, "recovery_policy_id": "RP_wait",
             "recovery_live_execution": True}
    assert recovery_cli_arguments(pilot) == ["--recovery-policy", "RP_wait", "--recovery-live-execution"]
    launch = recovery_launch_arguments("R3", goal={"x": 1.0, "y": -2.0, "yaw": 0.25},
                                       selector_model="models/s.json", live_execution=True)
    assert launch[0] == "recovery_policy:=R3" and "recovery_selector_model:=models/s.json" in launch
    assert "recovery_live_execution:=true" in launch and "goal_yaw:=0.25" in launch
    keys = [item.split(":=")[0] for item in launch]
    assert keys == [name for name in RECOVERY_LAUNCH_ARGUMENTS if name in keys]
    assert set(RECOVERY_LAUNCH_ARGUMENTS) - set(keys) == {
        "recovery_model_dir", "recovery_calibrator", "recovery_live_evidence",
    }


def test_launch_arguments_never_carry_an_empty_value():
    # ``ros2 launch`` refuses ``key:=`` as malformed: the post-freeze form (no model
    # overrides) must omit those keys and rely on the declared launch defaults.
    launch = recovery_launch_arguments("R2", goal={"x": 0.0, "y": 0.0})
    assert all(item.split(":=", 1)[1] != "" for item in launch)
    assert "recovery_model_dir" not in " ".join(launch) and "recovery_live_evidence" not in " ".join(launch)
    override = recovery_launch_arguments(
        "R2", goal={"x": 0.0, "y": 0.0}, live_execution=True, evidence_override="/tmp/draft.yaml",
    )
    assert "recovery_live_evidence:=/tmp/draft.yaml" in override


def test_evidence_override_is_admitted_only_for_engineering_smokes(tmp_path):
    root = tmp_path
    (root / "configs").mkdir()
    signed = root / "configs/recovery_live_evidence.yaml"
    signed.write_text("frozen: true\n")
    draft = root / "configs/recovery_live_evidence.draft.yaml"
    draft.write_text("frozen: true\n")
    smoke_root = root / "data/raw_engineering_smoke"
    smoke_root.mkdir(parents=True)
    admitted = validate_evidence_override(
        "configs/recovery_live_evidence.draft.yaml", campaign_id="recovery_smoke_manual",
        output_root=smoke_root, root=root,
    )
    assert admitted == draft.resolve()
    with pytest.raises(ValueError, match="campaign-id"):
        validate_evidence_override(str(draft), campaign_id="manual", output_root=smoke_root, root=root)
    with pytest.raises(ValueError, match="output-root"):
        validate_evidence_override(str(draft), campaign_id="recovery_smoke_x",
                                   output_root=root / "data/raw", root=root)
    with pytest.raises(ValueError, match="signed evidence"):
        validate_evidence_override(str(signed), campaign_id="recovery_smoke_x",
                                   output_root=smoke_root, root=root)
    with pytest.raises(ValueError, match="does not exist"):
        validate_evidence_override("configs/missing.yaml", campaign_id="recovery_smoke_x",
                                   output_root=smoke_root, root=root)
    block = recovery_system_block("R2", root=root, run_id="run", live_execution_requested=True,
                                  smoke_unfrozen=False, evidence_override=str(draft))
    assert block["recovery_live_evidence_override"] == str(draft)
    assert recovery_system_block("R2", root=root, run_id="run", live_execution_requested=False,
                                 smoke_unfrozen=False)["recovery_live_evidence_override"] is None


def test_episode_runner_refuses_evidence_override_outside_engineering_smokes(tmp_path):
    draft = tmp_path / "draft.yaml"
    draft.write_text("frozen: true\n")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_research2_episode.py"),
         "--map", "dev_00", "--route", "dev_00_r0", "--family", "lidar_dropout",
         "--severity", "medium", "--seed", "1", "--recovery-policy", "R2",
         "--recovery-live-execution", "--recovery-evidence-override", str(draft),
         "--campaign-id", "manual", "--output-root", str(ROOT / "data/raw_engineering_smoke")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0 and "campaign-id must start with 'recovery_smoke_'" in result.stderr
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_research2_episode.py"),
         "--map", "dev_00", "--route", "dev_00_r0", "--family", "lidar_dropout",
         "--severity", "medium", "--seed", "1", "--recovery-policy", "R2",
         "--recovery-live-execution", "--recovery-evidence-override", str(draft),
         "--campaign-id", "recovery_smoke_manual", "--output-root", str(tmp_path / "raw")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0 and "output-root must be data/raw_engineering_smoke" in result.stderr


def test_episode_runner_and_launch_keep_the_default_path_untouched():
    runner = (ROOT / "scripts/run_research2_episode.py").read_text(encoding="utf-8")
    # Every recovery block is guarded by the explicit argument; nothing runs for R0.
    assert runner.count("if args.recovery_policy is not None:") == 2
    assert "if recovery_system is not None:" in runner
    assert 'default=None' in runner.split('"--recovery-policy"')[1].split(")")[0]
    launch = (ROOT / "ros_ws/src/failure_experiment/launch/shared_sim.launch.py").read_text()
    assert 'DeclareLaunchArgument("recovery_policy", default_value="R0")' in launch
    assert "*recovery_nodes()" in launch and "*recovery_arguments()" in launch
    help_text = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_research2_episode.py"), "--help"],
        capture_output=True, text=True,
    ).stdout
    assert "--recovery-policy" in help_text and "RP_backup" in help_text


def test_launch_conditions_skip_monitor_and_manager_for_r0():
    launch = pytest.importorskip("launch")
    spec = importlib.util.spec_from_file_location(
        "shared_sim_launch", ROOT / "ros_ws/src/failure_experiment/launch/shared_sim.launch.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    nodes = [entity for entity in description.entities if entity.__class__.__name__ == "Node"]
    recovery = [node for node in nodes if node._Node__package in {"failure_monitor", "recovery_manager"}]
    assert len(recovery) == 2
    defaults = {
        argument.name: argument.default_value[0].text
        for argument in description.entities if isinstance(argument, launch.actions.DeclareLaunchArgument)
        and argument.name.startswith("recovery_") and argument.default_value
    }
    assert defaults["recovery_policy"] == "R0" and defaults["recovery_live_execution"] == "false"
    for policy, expected in (("R0", False), ("R2", True), ("RP_backup", True)):
        context = launch.LaunchContext()
        context.launch_configurations["recovery_policy"] = policy
        assert all(node.condition.evaluate(context) is expected for node in recovery)


def test_runner_follows_the_manager_resumed_goal_after_a_live_cancel():
    own = b"own-goal"
    # Our goal cancelled, nothing resumed yet: wait within the grace period.
    assert resumed_mission_terminal([(own, 10.0, 5)], own, now=100.0, last_activity=100.0) == (None, 100.0)
    # Resumed goal executing: activity refreshes; succeeded -> success; aborted -> planner_failure.
    assert resumed_mission_terminal([(own, 10.0, 5), (b"r1", 12.0, 2)], own, now=105.0,
                                    last_activity=100.0) == (None, 105.0)
    assert resumed_mission_terminal([(own, 10.0, 5), (b"r1", 12.0, 4)], own, now=120.0,
                                    last_activity=105.0) == ("success", 120.0)
    assert resumed_mission_terminal([(b"r1", 12.0, 6)], own, now=120.0,
                                    last_activity=105.0) == ("planner_failure", 120.0)
    # The newest goal decides even when an older resumed goal already succeeded or was cancelled.
    assert resumed_mission_terminal([(b"r1", 12.0, 5), (b"r2", 14.0, 2)], own, now=121.0,
                                    last_activity=105.0) == (None, 121.0)
    # A second intervention cancelled the resumed goal: wait, then time out after the grace.
    assert resumed_mission_terminal([(b"r1", 12.0, 5)], own, now=110.0, last_activity=105.0) == (None, 105.0)
    assert resumed_mission_terminal([(b"r1", 12.0, 5)], own, now=140.0, last_activity=105.0) == ("timeout", 105.0)
    assert resumed_mission_terminal([], own, now=131.0, last_activity=100.0, grace_seconds=30.0) == ("timeout", 100.0)


# ----------------------------------------------------------------------------- manager


def test_forced_pilot_action_obeys_guard_and_degrades_to_controlled_stop():
    request = RecoveryRequest("run", "w1", 0.9, "planning", state(rear_clearance_m=None))
    admitted = decide_recovery(request, GuardConfig(), "RP_spin_active_rescan")
    assert admitted["recommended_action"] == "spin_active_rescan"
    assert admitted["forced_action"] == "spin_active_rescan" and admitted["guard_rejected"] is False
    rejected = decide_recovery(request, GuardConfig(), "RP_backup")
    assert rejected["recommended_action"] == "controlled_stop" and rejected["guard_rejected"] is True
    assert rejected["forced_action"] == "backup"
    exhausted = decide_recovery(
        RecoveryRequest("run", "w2", 0.9, "planning", state(repeated_recovery_count=2)),
        GuardConfig(), "RP_backup",
    )
    assert exhausted["recommended_action"] == "request_assistance" and exhausted["abstained"]
    with pytest.raises(ValueError):
        decide_recovery(request, GuardConfig(), "RP_request_assistance")
    assert decide_recovery(request, GuardConfig(), "R2")["forced_action"] is None


class FakeCommander:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, *args))
        return record


def test_safe_stop_phase_runs_cancel_then_zero_and_controlled_stop_resumes():
    assert SAFE_STOP_PREFIX == ("cancel_task", "zero_cmd_vel")
    assert ACTION_SEQUENCES["controlled_stop"] == ("cancel_task", "zero_cmd_vel", "resume_navigation")
    assert all(seq[:2] == SAFE_STOP_PREFIX for seq in ACTION_SEQUENCES.values())
    events = []
    commander = FakeCommander()
    executor = LiveExecutor(commander, GateDecision(True, "ok"), events.append)
    record = executor.safe_stop("run", "w1", "R2")
    assert record["status"] == "executed" and [c[0] for c in commander.calls] == list(SAFE_STOP_PREFIX)
    assert events[-1]["event_type"] == "recovery_safe_stop"
    locked = LiveExecutor(FakeCommander(), GateDecision(False, "locked"), events.append)
    assert locked.safe_stop("run", "w1", "R2")["status"] == "refused_live_execution_locked"


# ----------------------------------------------------------------------------- summary block


def monitor_sidecar(alarms=1):
    return {"decision_count": 120, "engineering_smoke": False,
            "alarms": [{"decision_time": 30.5 + i, "risk_score": 0.8} for i in range(alarms)]}


def manager_sidecar(action="backup", executed=True, rejected=("relocalise",), forced=None,
                    guard_rejected=False):
    guard_results = {name: {"eligible": name not in rejected, "reason": "r"} for name in (
        "controlled_stop", "relocalise", "replan_clear_costmaps", "backup", "spin_active_rescan",
        "wait", "request_assistance",
    )}
    return {"gate": {"enabled": executed}, "decisions": [{
        "warning_id": "run-w001", "decision_time": 30.5, "recommended_action": action,
        "forced_action": forced, "guard_rejected": guard_rejected, "guard_results": guard_results,
        "execution": {"status": "executed" if executed else "refused_live_execution_locked",
                      "execution_performed": executed, "seconds": 2.5},
    }]}


def test_summary_block_reports_warnings_actions_and_guard_rejections(tmp_path):
    summary = summarise_recovery_sidecars(monitor_sidecar(2), manager_sidecar())
    assert summary["warnings_issued"] == 2 and summary["first_warning_time"] == 30.5
    assert summary["intervention_count"] == 1 and summary["actions"][0]["action"] == "backup"
    assert summary["guard_rejections"][0]["rejected_actions"] == ["relocalise"]
    assert summary["guard_violation"] is False and summary["execution_results"][0]["failed"] is False
    root = tmp_path
    (root / "logs/failure-monitor").mkdir(parents=True)
    (root / "logs/recovery").mkdir(parents=True)
    (root / "logs/failure-monitor/run.partial.json").write_text(json.dumps(monitor_sidecar()))
    (root / "logs/recovery/run.json").write_text(json.dumps(manager_sidecar()))
    block = recovery_system_block("RP_backup", root=root, run_id="run",
                                  live_execution_requested=True, smoke_unfrozen=False)
    assert block["recovery_forced_action"] == "backup" and block["online_failure_monitor"] is True
    assert block["recovery"]["monitor_sidecar_present"] is True   # partial promoted
    assert (root / "logs/failure-monitor/run.json").exists()
    r0 = recovery_system_block("R0", root=root, run_id="other", live_execution_requested=False,
                               smoke_unfrozen=False)
    assert r0["recovery"] is None and r0["online_failure_monitor"] is False


def test_guard_violation_detected_when_an_ineligible_action_ran():
    tampered = manager_sidecar(action="relocalise", rejected=("relocalise",))
    assert summarise_recovery_sidecars(monitor_sidecar(), tampered)["guard_violation"] is True


# ----------------------------------------------------------------------------- campaigns


def splits():
    return yaml.safe_load((ROOT / "data/manifests/splits.template.yaml").read_text())


def pilot_document():
    return build_manifest(validation_map_routes(splits()), seed_base=2_000_000, gate={"frozen": False})


def test_recovery_pilot_design_is_882_paired_validation_episodes():
    document = pilot_document()
    assert validate_recovery_pilot(document, splits()) == []
    assert document["expected_base_episode_count"] == 126
    assert document["expected_episode_count"] == 882
    assert document["allowed_splits"] == ["validation"] and document["protected_test_used"] is False
    assert document["execution_policy"]["concurrency"] == 6
    assert document["parallel_execution_admitted_by"] == "PA-2026-09-04-02"
    base = targeted_execution_order(expand_balanced_pilot(document))
    episodes = expand_recovery_policies(document, base)
    assert len(episodes) == 882 and len({e["episode_key"] for e in episodes}) == 882
    assert all(e["map"].startswith("val_") for e in episodes)
    assert [e["recovery_policy_id"] for e in episodes[:7]] == list(PILOT_POLICY_SET)
    assert len({e["pair_key"] for e in episodes[:7]}) == 1
    assert episodes[1]["recovery_live_execution"] is True and "recovery_live_execution" not in episodes[0]
    validation = yaml.safe_load((ROOT / "data/manifests/balanced_validation_v1.yaml").read_text())
    assert {e["seed"] for e in episodes}.isdisjoint(e["seed"] for e in expand_balanced_pilot(validation))


def test_recovery_pilot_validator_rejects_wrong_policies_split_and_concurrency():
    document = pilot_document()
    document["recovery_policies"] = ["R0", "R2"]
    document["allowed_splits"] = ["development"]
    document["execution_policy"]["concurrency"] = 1
    document["parallel_execution_admitted_by"] = None
    findings = validate_recovery_pilot(document, splits())
    assert any("six forced actions" in f for f in findings)
    assert any("validation maps only" in f for f in findings)
    assert any("concurrency 6" in f for f in findings)
    assert any("PA-2026-09-04-02" in f for f in findings)


def test_plain_manifests_expand_unchanged():
    document = yaml.safe_load((ROOT / "data/manifests/balanced_pilot_v1.yaml").read_text())
    base = expand_balanced_pilot(document)
    assert expand_recovery_policies(document, base) == base
    assert all(recovery_cli_arguments(item) == [] for item in base)


def test_pilot_manifest_cli_dry_run_never_writes(tmp_path):
    script = ROOT / "scripts/build_recovery_pilot_manifest.py"
    result = subprocess.run([sys.executable, str(script), "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '"episodes": 882' in result.stdout
    output = tmp_path / "pilot.yaml"
    written = subprocess.run([sys.executable, str(script), "--output", str(output)],
                             capture_output=True, text=True)
    assert written.returncode == 0, written.stderr
    assert yaml.safe_load(output.read_text())["campaign_id"] == "recovery_pilot_v1"


# ----------------------------------------------------------------------------- outcome and cost tables


def fake_summary(run_id, policy, *, cell=0, success=True, collision=False, duration=60.0,
                 path=10.0, action=None, guard_rejected=False, forced=None, executed=True,
                 warnings=1, failed=False, campaign="recovery_pilot_v1", split="validation",
                 manager_sidecar_path=None):
    recovery = None
    if policy != "R0":
        actions = [] if action is None else [{
            "warning_id": f"{run_id}-w001", "decision_time": 30.5, "action": action,
            "forced_action": forced, "guard_rejected": guard_rejected,
            "execution_status": "failed" if failed else "executed", "execution_performed": executed,
            "seconds": 2.0,
        }]
        recovery = {
            "warnings_issued": warnings, "first_warning_time": 30.5 if warnings else None,
            "decisions_scored": 100, "engineering_smoke": False, "actions": actions,
            "guard_rejections": [], "execution_results": [
                {"warning_id": a["warning_id"], "action": a["action"], "status": a["execution_status"],
                 "failed": a["execution_status"] == "failed"} for a in actions],
            "intervention_count": sum(1 for a in actions if a["execution_performed"]),
            "guard_violation": False, "live_execution_enabled": True,
            "monitor_sidecar_present": True, "manager_sidecar_present": True,
        }
    return {
        "identity": {"run_id": run_id, "campaign_id": campaign, "episode_key": f"cell{cell}-{policy}"},
        "environment": {"map_id": "val_00", "route_id": f"val_00_r{cell % 6}", "seed": 2_000_000 + cell,
                        "split": split, "protected_test_used": False},
        "label_only": {"fault_family": "lidar_dropout", "severity": "medium", "recovery_policy_id": policy},
        "outcome": {"terminal_state": "success" if success else "timeout", "success": success,
                    "collision": collision, "duration_s": duration, "path_length_m": path},
        "system": {"recovery_policy_id": policy, "recovery_forced_action": forced,
                   "manager_sidecar": manager_sidecar_path, "recovery": recovery},
    }


def test_outcome_rows_pair_against_r0_and_compute_regret():
    weights = load_cost_weights()
    summaries = [
        fake_summary("a", "R0", duration=60.0, path=10.0),
        fake_summary("b", "RP_backup", duration=66.0, path=11.5, action="backup", forced="backup"),
        fake_summary("c", "RP_wait", duration=70.0, path=10.0, action="wait", forced="wait",
                     success=False),
        fake_summary("d", "RP_relocalise", cell=1, action="controlled_stop", forced="relocalise",
                     guard_rejected=True),        # no R0 reference for cell 1
        fake_summary("e", "RP_spin_active_rescan", success=False, action="spin_active_rescan",
                     forced="spin_active_rescan"),
    ]
    summaries[-1]["outcome"]["terminal_state"] = "invalid"
    rows, report = build_outcome_rows(summaries, campaign_id="recovery_pilot_v1",
                                      expected_policies=list(PILOT_POLICY_SET), weights=weights)
    assert report["pairs"] == 1 and report["pairs_without_reference"] == [["val_00", "val_00_r1", "2000001", "lidar_dropout", "medium"]]
    assert [r["reason"] for r in report["excluded"]] == ["invalid episode"]
    by_policy = {row["policy_id"]: row for row in rows}
    assert set(by_policy) == {"R0", "RP_backup", "RP_wait"}
    assert by_policy["R0"]["added_time_seconds"] == "0.0" and by_policy["R0"]["intervention_count"] == "0"
    assert by_policy["RP_backup"]["added_time_seconds"] == "6.0"
    assert by_policy["RP_backup"]["added_path_length_m"] == "1.5"
    assert by_policy["RP_backup"]["unnecessary_intervention"] == "true"   # R0 completed
    assert by_policy["RP_wait"]["mission_complete"] == "false"
    assert float(by_policy["R0"]["action_regret_vs_oracle"]) == 0.0
    assert float(by_policy["RP_wait"]["action_regret_vs_oracle"]) > float(by_policy["RP_backup"]["action_regret_vs_oracle"]) > 0
    assert by_policy["RP_wait"]["oracle_action"] == "none"
    payload = rows_to_csv_bytes(rows)
    header = payload.decode().splitlines()[0].split(",")
    for column in ("map_id", "route_id", "seed", "fault_family", "severity", "policy_id",
                   "mission_complete", "collision", "guard_violation", "guard_rejected",
                   "added_time_seconds", "added_path_length_m", "intervention_count",
                   "recovery_action", "action_regret_vs_oracle"):
        assert column in header


def test_cost_rows_use_recorded_state_and_never_train_on_rejected_actions(tmp_path):
    weights = load_cost_weights()
    summaries = [fake_summary("r0", "R0")]
    sidecars = {}
    manager_state = state(rear_clearance_m=None, localisation_poor=True, relocalisation_available=True)
    guards = eligible_actions(manager_state, GuardConfig())
    for index, policy in enumerate(PILOT_POLICY_IDS):
        forced = forced_action(policy)
        rejected = not guards[forced][0]
        executed = "controlled_stop" if rejected else forced
        run_id = f"p{index}"
        summaries.append(fake_summary(run_id, policy, action=executed, forced=forced,
                                      guard_rejected=rejected, duration=62.0 + index))
        sidecars[run_id] = {"decisions": [{
            "warning_id": f"{run_id}-w001", "risk_score": 0.83, "diagnosed_signal_group": "localisation",
            "recommended_action": executed, "forced_action": forced, "guard_rejected": rejected,
            "guard_results": {name: {"eligible": ok, "reason": why} for name, (ok, why) in guards.items()},
            "state": manager_state.__dict__,
        }]}
    outcomes, _report = build_outcome_rows(summaries, campaign_id="recovery_pilot_v1",
                                           expected_policies=list(PILOT_POLICY_SET), weights=weights)
    rows, report = cost_rows(outcomes, {s["identity"]["run_id"]: s for s in summaries}, sidecars, GuardConfig())
    assert report["guard_disagreements"] == [] and report["reference_rows_skipped"] == 1
    backup = [row for row in rows if row["candidate_action"] == "backup"]
    assert backup and backup[0]["guard_eligible"] == "false"      # rear clearance unverified
    stops = [row for row in rows if row["candidate_action"] == "controlled_stop"]
    rejected_forced = sum(1 for action in ("relocalise", "replan_clear_costmaps", "backup",
                                           "spin_active_rescan", "wait") if not guards[action][0])
    assert rejected_forced >= 1 and len(stops) == 1 + rejected_forced   # forced + fallbacks
    assert all(row["split"] == "validation" for row in rows) and len({row["warning_id"] for row in rows}) == 1
    assert all(row["localisation_poor"] == "true" and row["rear_clearance_m"] == "" for row in rows)
    # The trainer refuses the validation rows without the amendment and never fits rejected rows.
    with pytest.raises(ValueError, match="amendment"):
        train_selector(rows, GuardConfig(), fit_split="validation", minimum_rows_per_action=1)
    model = train_selector(rows, GuardConfig(), fit_split="validation",
                           fit_split_admitted_by="PA-2026-09-03-04", minimum_rows_per_action=1)
    assert "backup" not in model["actions"] and "relocalise" in model["actions"]
    assert model["fit_split"] == "validation" and model["fit_split_admitted_by"] == "PA-2026-09-03-04"
    with pytest.raises(ValueError, match="held-out"):
        train_selector(rows, GuardConfig(), fit_split="held_out_map_test")


def test_selector_cli_accepts_validation_fit_only_with_amendment(tmp_path):
    rows = []
    for index in range(12):
        rows.append({
            "split": "validation", "map_id": "val_00", "route_id": "val_00_r0", "seed": str(index),
            "fault_family": "lidar_dropout", "severity": "medium", "warning_id": f"w{index}",
            "risk_score": "0.8", "diagnosed_signal_group": "planning", "stopped": "true",
            "stop_allowed": "true", "localisation_poor": "false", "planning_stale_or_blocked": "true",
            "rear_clearance_m": "1.0", "rotation_clearance_m": "1.0", "immediate_collision_risk": "false",
            "obstruction_may_be_transient": "true", "relocalisation_available": "false",
            "repeated_recovery_count": "0", "candidate_action": "replan_clear_costmaps",
            "guard_eligible": "true", "observed_cost": str(3.0 + index),
        })
    table = tmp_path / "costs.csv"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    script = ROOT / "scripts/train_recovery_selector.py"
    refused = subprocess.run([sys.executable, str(script), str(table), "--output", str(tmp_path / "a.json"),
                              "--fit-split", "validation"], capture_output=True, text=True)
    assert refused.returncode != 0 and "amendment" in refused.stderr
    bogus = subprocess.run([sys.executable, str(script), str(table), "--output", str(tmp_path / "b.json"),
                            "--fit-split", "validation", "--amendment-id", "PA-2026-01-01-99"],
                           capture_output=True, text=True)
    assert bogus.returncode != 0 and "not on file" in bogus.stderr
    accepted = subprocess.run([sys.executable, str(script), str(table), "--output", str(tmp_path / "c.json"),
                               "--fit-split", "validation", "--amendment-id", "PA-2026-09-03-04"],
                              capture_output=True, text=True)
    assert accepted.returncode == 0, accepted.stderr
    model = json.loads((tmp_path / "c.json").read_text())
    assert model["training_splits_used"] == ["validation"]
    assert model["inputs"]["amendment"]["amendment_id"] == "PA-2026-09-03-04"


def test_outcome_table_cli_end_to_end(tmp_path):
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    for run_id, policy, kwargs in (
        ("a", "R0", {}), ("b", "RP_backup", {"action": "backup", "forced": "backup", "duration": 65.0}),
    ):
        (summaries / f"{run_id}.yaml").write_text(yaml.safe_dump(fake_summary(run_id, policy, **kwargs)))
    manifest = tmp_path / "pilot.yaml"
    manifest.write_text(yaml.safe_dump(pilot_document()))
    output = tmp_path / "outcomes.csv"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/build_recovery_outcome_table.py"),
        "--manifest", str(manifest), "--summaries", str(summaries), "--output", str(output),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(output.open()))
    assert {row["policy_id"] for row in rows} == {"R0", "RP_backup"}
    provenance = json.loads(output.with_name("outcomes.csv.provenance.json").read_text())
    assert provenance["pairs"] == 1 and re.fullmatch(r"[0-9a-f]{64}", provenance["output_sha256"])
    again = subprocess.run([
        sys.executable, str(ROOT / "scripts/build_recovery_outcome_table.py"),
        "--manifest", str(manifest), "--summaries", str(summaries), "--output", str(output),
    ], capture_output=True, text=True)
    assert again.returncode != 0

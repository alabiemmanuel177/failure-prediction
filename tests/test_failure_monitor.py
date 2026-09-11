"""Online failure monitor: replay equivalence with the offline pipeline, fail-closed start."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ros_ws/src/failure_monitor"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from failure_monitor import monitor_core as core  # noqa: E402
from src.evaluation.policy import AlarmPolicy, apply_alarm_policy  # noqa: E402
from src.features import (  # noqa: E402
    LeakagePolicy, load_primary_feature_set, load_raw_feature_contract, model_columns,
)
from src.features.sequences import _grid  # noqa: E402
from src.recovery.plumbing import PILOT_POLICY_IDS  # noqa: E402


DATASET = ROOT / "data/derived/balanced_pilot_v1-development-648"
MODEL_DIR = ROOT / "models/preliminary_v1/p3_causal_tcn__seed20260903"
CALIBRATOR = ROOT / "reports/calibration/preliminary_v1/p3_causal_tcn.calibrator.json"
SCHEMA = ROOT / "configs/feature_schema.yaml"


def first_episode():
    manifest = DATASET / "extraction_manifest.jsonl"
    if not manifest.exists():
        pytest.skip("development derived dataset is not available")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("protected_test_used") is False and row.get("split") == "development" \
                and (DATASET / "decisions" / f"{row['run_id']}.npz").exists():
            return row
    pytest.skip("no development episode with a decisions artifact")


def episode_inputs(row):
    run_id = row["run_id"]
    telemetry = list(csv.DictReader((DATASET / "telemetry" / f"{run_id}.csv").open(newline="")))
    annotation = yaml.safe_load((DATASET / "annotations" / f"{run_id}.yaml").read_text())
    summary = yaml.safe_load((ROOT / "data/raw/summaries" / f"{run_id}.yaml").read_text())
    with np.load(DATASET / "decisions" / f"{run_id}.npz", allow_pickle=False) as artifact:
        decisions = {name: artifact[name] for name in artifact.files}
    return run_id, telemetry, annotation, summary, decisions


def replay_windows(run_id, telemetry, annotation, summary, decisions, *, until_time=None):
    """Feed recorded samples in receive order and collect every complete live window."""
    contract = load_raw_feature_contract(SCHEMA)
    primary = load_primary_feature_set(SCHEMA)
    goal = summary["environment"]["goal_pose"]
    builder = core.WindowBuilder(
        run_id=run_id, contract=contract, primary_features=primary,
        leakage_policy=LeakagePolicy.from_yaml(ROOT / "configs/leakage_denylist.yaml"),
        goal_x=float(goal["x"]), goal_y=float(goal["y"]),
    )
    builder.start(float(annotation["episode"]["start_time"]))
    samples = sorted(telemetry, key=lambda item: float(item["timestamp"]))
    cursor = 0
    end = float(annotation["episode"]["end_time"]) if until_time is None else until_time
    windows = {}
    for grid_index, moment in builder.due_grid_times(end):
        while cursor < len(samples) and float(samples[cursor]["timestamp"]) <= moment:
            item = samples[cursor]
            builder.buffer.add(item["feature"], item["source"], float(item["timestamp"]), float(item["value"]))
            cursor += 1
        window = builder.advance(grid_index, moment)
        if window is not None:
            windows[builder.decision_index(grid_index)] = (moment, window)
    return builder, windows


def test_grid_matches_offline_sampling():
    start, stride = 13.193, 0.5
    offline = _grid(start, 20.0, stride)
    assert [core.grid_time(start, stride, k) for k in range(1, len(offline) + 1)] == offline


def test_replay_windows_equal_offline_decisions():
    row = first_episode()
    run_id, telemetry, annotation, summary, decisions = episode_inputs(row)
    builder, windows = replay_windows(run_id, telemetry, annotation, summary, decisions)
    assert tuple(decisions["feature_names"].tolist()) == builder.columns
    compared = 0
    worst = 0.0
    eligible = 0
    for position, index in enumerate(decisions["decision_index"].tolist()):
        moment, window = windows[int(index)]
        assert moment == float(decisions["decision_time"][position])
        difference = float(np.max(np.abs(window - decisions["X"][position])))
        worst = max(worst, difference)
        assert difference <= 1e-6, f"decision {index}: max |diff| {difference}"
        compared += 1
        eligible += int(decisions["y"][position] >= 0)
    assert compared == decisions["X"].shape[0] > 0 and eligible > 0
    assert worst <= 1e-6
    assert builder.buffer.dropped_out_of_order == 0 and builder.buffer.dropped_unknown == 0


def test_suffix_enrichment_equals_full_recomputation():
    row = first_episode()
    run_id, telemetry, annotation, summary, decisions = episode_inputs(row)
    builder, windows = replay_windows(run_id, telemetry, annotation, summary, decisions)
    # The live builder discards rows older than the exact suffix it needs.
    assert len(builder.raw_rows) <= builder.steps + 1 + 40
    assert len(windows) == decisions["X"].shape[0]


def test_sample_buffer_rejects_denylisted_out_of_order_and_mixed_sources():
    contract = load_raw_feature_contract(SCHEMA)
    buffer = core.SampleBuffer(contract)
    assert buffer.add("command_linear", "/cmd_vel", 1.0, 0.1)
    assert not buffer.add("command_linear", "/cmd_vel", 0.5, 0.2)
    assert buffer.dropped_out_of_order == 1
    assert not buffer.add("command_linear", "/odom", 2.0, 0.2) and buffer.dropped_unknown == 1
    assert buffer.add("confidence_mean", "/research2/features/perception", 1.0, 0.9)
    assert not buffer.add("confidence_mean", "/semantic/confidence", 2.0, 0.9)
    assert buffer.dropped_mixed_source == 1
    assert not buffer.add("ground_truth_x", "/ground_truth_pose", 1.0, 0.0)
    buffer.add("command_linear", "/cmd_vel", 1.2, 0.3)
    buffer.add("command_linear", "/cmd_vel", 5.0, 0.3)
    buffer.prune(6.0)
    assert [s.timestamp for s in buffer.samples["command_linear"]] == [5.0]


def test_leakage_denylist_refuses_event_topic_subscriptions():
    policy = LeakagePolicy.from_yaml(ROOT / "configs/leakage_denylist.yaml")
    core.check_subscriptions(policy)          # the frozen subscription list is clean
    with pytest.raises(ValueError):
        core.check_subscriptions(policy, ("/research2/events",))
    with pytest.raises(ValueError):
        core.check_subscriptions(policy, ("/ground_truth_pose",))
    assert "/research2/events" not in core.SUBSCRIBED_TOPICS


def test_monitor_refuses_without_freeze_unless_engineering_smoke(tmp_path):
    alarm = tmp_path / "alarm_policy.yaml"
    alarm.write_text(yaml.safe_dump({"threshold": None, "persistence": {
        "required_above_threshold": 2, "decisions_considered": 3}, "cooldown_seconds": 10.0}))
    freeze = tmp_path / "model_freeze.yaml"
    with pytest.raises(core.MonitorRefused, match="not frozen"):
        core.resolve_monitor_assets(ROOT, freeze_path=freeze, alarm_policy_path=alarm, smoke_unfrozen=False)
    with pytest.raises(core.MonitorRefused, match="explicit model_dir"):
        core.resolve_monitor_assets(ROOT, freeze_path=freeze, alarm_policy_path=alarm, smoke_unfrozen=True)
    if not (MODEL_DIR / "checkpoint.pt").exists() or not CALIBRATOR.exists():
        pytest.skip("preliminary model or calibrator absent")
    assets = core.resolve_monitor_assets(
        ROOT, freeze_path=freeze, alarm_policy_path=alarm, smoke_unfrozen=True,
        model_dir_override=str(MODEL_DIR), calibrator_override=str(CALIBRATOR), smoke_threshold=0.4,
    )
    assert assets.engineering_smoke is True and assets.alarm_policy.threshold == 0.4
    assert assets.threshold_source == "engineering_smoke_parameter" and assets.frozen is False
    # A frozen record pins both hashes and refuses overrides.
    freeze.write_text(yaml.safe_dump({
        "frozen": True,
        "predictor": {"checkpoint": str((MODEL_DIR / "checkpoint.pt").relative_to(ROOT)),
                      "checkpoint_sha256": hashlib.sha256((MODEL_DIR / "checkpoint.pt").read_bytes()).hexdigest()},
        "calibration": {"artifact": str(CALIBRATOR.relative_to(ROOT)),
                        "artifact_sha256": hashlib.sha256(CALIBRATOR.read_bytes()).hexdigest()},
    }))
    with pytest.raises(core.MonitorRefused, match="no frozen threshold"):
        core.resolve_monitor_assets(ROOT, freeze_path=freeze, alarm_policy_path=alarm, smoke_unfrozen=False)
    alarm.write_text(yaml.safe_dump({"threshold": 0.31, "persistence": {
        "required_above_threshold": 2, "decisions_considered": 3}, "cooldown_seconds": 10.0}))
    frozen = core.resolve_monitor_assets(ROOT, freeze_path=freeze, alarm_policy_path=alarm, smoke_unfrozen=False)
    assert frozen.frozen is True and frozen.engineering_smoke is False and frozen.alarm_policy.threshold == 0.31
    with pytest.raises(core.MonitorRefused, match="overrides are refused"):
        core.resolve_monitor_assets(ROOT, freeze_path=freeze, alarm_policy_path=alarm, smoke_unfrozen=False,
                                    model_dir_override=str(tmp_path))
    tampered = yaml.safe_load(freeze.read_text())
    tampered["predictor"]["checkpoint_sha256"] = "0" * 64
    freeze.write_text(yaml.safe_dump(tampered))
    with pytest.raises(core.MonitorRefused, match="checkpoint"):
        core.resolve_monitor_assets(ROOT, freeze_path=freeze, alarm_policy_path=alarm, smoke_unfrozen=False)


def test_signal_group_diagnosis_uses_largest_observed_contribution():
    names = model_columns(load_primary_feature_set(SCHEMA))
    groups = core.load_feature_groups(ROOT / "configs/ablations.yaml")
    step = np.zeros(len(names), dtype=np.float32)
    index = {name: position for position, name in enumerate(names)}
    step[index["pose_covariance_trace"]] = 4.0
    step[index["minimum_front_range"]] = -1.0
    group, contributions = core.diagnose_signal_group(step, names, groups)
    assert group == "localisation" and contributions["localisation"] == 2.0
    step[index["pose_covariance_trace__missing"]] = 1.0     # masked -> not counted
    step[index["pose_jump__missing"]] = 1.0
    assert core.diagnose_signal_group(step, names, groups)[0] == "frontal_blockage"
    assert core.diagnose_signal_group(np.zeros(len(names)), names, groups)[0] == "unknown"
    assert set(core.GROUP_TO_SIGNAL) == set(groups)


def test_state_estimate_and_scan_clearances():
    clearances = core.scan_clearances([1.0, 0.5, float("inf"), 0.2], -np.pi, np.pi / 2)
    assert clearances["rear_clearance_m"] == 1.0 and clearances["rotation_clearance_m"] == 0.2
    row = {
        "decision_time": 30.0, "measured_linear": 0.0, "measured_linear__missing": 0,
        "measured_linear__age_seconds": 0.1, "measured_angular": 0.0, "measured_angular__missing": 0,
        "pose_covariance_trace": 0.9, "pose_covariance_trace__missing": 0,
        "global_path_length": 3.0, "global_path_length__missing": 0, "replan_rate": 0.0,
        "replan_rate__missing": 0, "minimum_front_range": 0.6, "minimum_front_range__missing": 0,
    }
    state = core.estimate_robot_state(
        row, diagnosed="frontal_blockage", aux={"rear_clearance_m": (29.9, 0.8), "rotation_clearance_m": (20.0, 0.5)},
        now=30.0, config=core.StateEstimatorConfig(), relocalisation_available=False, repeated_recovery_count=1,
    )
    assert state.stopped is True and state.localisation_poor is True
    assert state.rear_clearance_m == 0.8 and state.rotation_clearance_m is None   # stale scan
    assert state.obstruction_may_be_transient is True and state.immediate_collision_risk is False
    assert state.repeated_recovery_count == 1 and state.stop_allowed is True


class FakeScorer:
    """Deterministic stand-in: risk = fraction of decisions seen, alarm rule replayed."""

    def __init__(self, policy: AlarmPolicy, feature_names):
        self.policy = policy
        self.feature_names = tuple(feature_names)
        self.rows = []

    def score(self, window, decision_time):
        risk = min(1.0, 0.1 * (len(self.rows) + 1))
        self.rows.append({"decision_time": decision_time, "risk_score": risk})
        last = apply_alarm_policy(self.rows, self.policy)[-1]
        return risk, risk, bool(last["persistent"]), bool(last["alarm"]), window


def test_monitor_core_issues_requests_with_forced_action_and_writes_sidecar(tmp_path):
    row = first_episode()
    run_id, telemetry, annotation, summary, decisions = episode_inputs(row)
    policy = AlarmPolicy(0.35, 2, 3, 10.0)
    assets = core.MonitorAssets(
        model_dir=tmp_path, calibrator_path=tmp_path / "c.json", checkpoint_sha256="0" * 64,
        calibrator_sha256="1" * 64, alarm_policy=policy, threshold_source="test",
        engineering_smoke=True, freeze_path=None, freeze_sha256=None, frozen=False,
    )
    goal = summary["environment"]["goal_pose"]
    monitor = core.MonitorCore(run_id=run_id, policy_id="RP_backup", assets=assets, root=ROOT,
                               goal_x=float(goal["x"]), goal_y=float(goal["y"]))
    monitor.scorer = FakeScorer(policy, model_columns(load_primary_feature_set(SCHEMA)))
    monitor.mission_started(float(annotation["episode"]["start_time"]))
    samples = sorted(telemetry, key=lambda item: float(item["timestamp"]))
    end = float(annotation["episode"]["start_time"]) + 20.0
    for item in samples:
        if float(item["timestamp"]) <= end:
            monitor.receive_scan_clearances([0.9, 0.9, 0.9], -np.pi, np.pi, float(item["timestamp"]))
            monitor.windows.buffer.add(item["feature"], item["source"], float(item["timestamp"]), float(item["value"]))
    outputs = monitor.step(end)
    alarms = [output for output in outputs if output.alarm]
    assert outputs[0].decision_index == 0 and len(alarms) >= 1
    assert alarms[0].warning_id == f"{run_id}-w001" and alarms[0].state is not None
    values = monitor.request_values(alarms[0])
    assert values["forced_action"] == "backup" and values["policy_id"] == "RP_backup"
    assert values["engineering_smoke"] == "true" and values["stopped"] in {"true", "false"}
    assert values["rear_clearance_m"] == "0.9" and values["rotation_clearance_m"] == "0.9"
    # Cooldown: consecutive alarms are at least 10 s apart.
    times = [output.decision_time for output in alarms]
    assert all(b - a >= 10.0 for a, b in zip(times, times[1:]))
    sidecar = monitor.sidecar()
    core.write_sidecar(sidecar, tmp_path / "m.json")
    reloaded = json.loads((tmp_path / "m.json").read_text())
    assert reloaded["alarm_count"] == len(alarms) and reloaded["engineering_smoke"] is True
    assert reloaded["decision_count"] == len(outputs) and reloaded["causal_role"] == "label_only"
    assert reloaded["compute_ms"]["p95"] is not None


def test_pilot_policies_are_accepted_by_the_core(tmp_path):
    assets = core.MonitorAssets(
        model_dir=tmp_path, calibrator_path=tmp_path, checkpoint_sha256="", calibrator_sha256="",
        alarm_policy=AlarmPolicy(0.5), threshold_source="t", engineering_smoke=True,
        freeze_path=None, freeze_sha256=None, frozen=False,
    )
    for policy in PILOT_POLICY_IDS:
        monitor = core.MonitorCore(run_id="r", policy_id=policy, assets=assets, root=ROOT, goal_x=0, goal_y=0)
        assert monitor.forced_action == policy[3:]
    with pytest.raises(ValueError):
        core.MonitorCore(run_id="r", policy_id="R7", assets=assets, root=ROOT, goal_x=0, goal_y=0)


def test_frozen_scorer_reproduces_offline_scores_and_alarms(tmp_path):
    pytest.importorskip("torch")
    if not (MODEL_DIR / "checkpoint.pt").exists() or not CALIBRATOR.exists():
        pytest.skip("preliminary model or calibrator absent")
    row = first_episode()
    run_id, telemetry, annotation, summary, decisions = episode_inputs(row)
    alarm = tmp_path / "alarm_policy.yaml"
    alarm.write_text(yaml.safe_dump({"threshold": None, "persistence": {
        "required_above_threshold": 2, "decisions_considered": 3}, "cooldown_seconds": 10.0}))
    assets = core.resolve_monitor_assets(
        ROOT, freeze_path=tmp_path / "absent.yaml", alarm_policy_path=alarm, smoke_unfrozen=True,
        model_dir_override=str(MODEL_DIR), calibrator_override=str(CALIBRATOR), smoke_threshold=0.2,
    )
    scorer = core.Scorer(assets, core.load_feature_groups(ROOT / "configs/ablations.yaml"))
    goal = summary["environment"]["goal_pose"]
    monitor = core.MonitorCore(run_id=run_id, policy_id="R2", assets=assets, root=ROOT,
                               goal_x=float(goal["x"]), goal_y=float(goal["y"]), scorer=scorer)
    monitor.mission_started(float(annotation["episode"]["start_time"]))
    limit = 60
    samples = sorted(telemetry, key=lambda item: float(item["timestamp"]))
    cursor = 0
    outputs = []
    for grid_index, moment in monitor.windows.due_grid_times(float(annotation["episode"]["end_time"])):
        while cursor < len(samples) and float(samples[cursor]["timestamp"]) <= moment:
            item = samples[cursor]
            monitor.windows.buffer.add(item["feature"], item["source"], float(item["timestamp"]), float(item["value"]))
            cursor += 1
        outputs.extend(monitor.step(moment))
        if len(outputs) >= limit:
            break
    assert len(outputs) == limit
    # Offline reference: the same checkpoint on the stored windows, then the same policy.
    prepared = scorer.model.prepare(decisions["X"][:limit])
    raw = scorer.model.score_prepared(prepared)
    risk = scorer.calibrator.apply(raw)
    live_raw = np.asarray([output.raw_score for output in outputs])
    live_risk = np.asarray([output.risk_score for output in outputs])
    assert np.max(np.abs(live_raw - raw)) <= 1e-5 and np.max(np.abs(live_risk - risk)) <= 1e-5
    offline_rows = apply_alarm_policy(
        [{"decision_time": float(t), "risk_score": float(r)} for t, r in zip(decisions["decision_time"][:limit], risk)],
        assets.alarm_policy,
    )
    assert [o.alarm for o in outputs] == [bool(r["alarm"]) for r in offline_rows]
    assert [o.persistent for o in outputs] == [bool(r["persistent"]) for r in offline_rows]
    assert all(o.diagnosed_signal_group in {*core.GROUP_TO_SIGNAL.values()} for o in outputs)
    assert monitor.sidecar()["compute_ms"]["median"] < 200.0

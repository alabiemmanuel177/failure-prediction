"""Synthetic tests for the Research 1 causal adapter and the supplement plan."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.experiments import expand_balanced_pilot, validate_balanced_pilot
from src.research1_adapter import (
    RESEARCH2_FAULT_FAMILIES, TERMINAL_EVENT_CLASS, AdapterThresholds, BagEvidence,
    ReceiveClockMap, RouteSpec, assess_episode, build_event_sidecar, distribute_blocks,
    episode_key, research2_fault_family, research2_severity, supplement_episode_count,
)
from scripts.plan_development_supplement import build_supplement, collision_findings


ROOT = Path(__file__).resolve().parents[1]
EVENT_CONFIG = yaml.safe_load((ROOT / "configs/failure_events.yaml").read_text(encoding="utf-8"))
ROUTE = RouteSpec(
    map_id="dev_01", route_id="dev_01_r0", start={"x": 4.0, "y": 0.0, "yaw": 0.0},
    goal={"x": -0.1, "y": -0.8}, shortest_path_m=4.9, goal_tolerance_m=0.25,
    episode_timeout_s=240.0,
)
CATALOG_ROW = {
    "run_id": "11111111-2222-3333-4444-555555555555", "map_id": "dev_01",
    "route_id": "dev_01_r0", "system_id": "S0", "seed": "3", "split": "development",
    "bag_path": "results/bags/x", "metadata_sha256": "0" * 64,
}


def aggregate(**overrides) -> dict:
    row = {
        "run_id": CATALOG_ROW["run_id"], "map_id": "dev_01", "route_id": "dev_01_r0",
        "system_id": "S0", "seed": "3", "shift_family": "lighting", "severity": "2",
        "terminal_state": "collision", "success": "False", "collision": "True",
        "timeout": "False", "invalid_reason": "", "duration_s": "20.0",
        "goal_distance_gt_m": "2.5", "commit_sha": "abc",
    }
    row.update({key: str(value) for key, value in overrides.items()})
    return row


def evidence(*, wall_start=1000.0, sim_start=10.0, rtf=0.8, seconds=40.0,
             contact_after_start=None, topics=None, bt=True,
             commands_until=None) -> BagEvidence:
    """Synthetic bag: wall-clock recorder, simulation headers, constant real-time factor."""
    knots, odometry, commands, truth, amcl = [], [], [], [], []
    step = 0.05
    n = int(seconds / step)
    for index in range(n + 1):
        wall = wall_start + index * step
        sim = sim_start + index * step * rtf
        knots.append((wall, sim))
        x = 4.0 - 0.1 * index * step * rtf
        odometry.append((sim, x, 0.0))
        commands.append((wall, 0.2))
        if index % 2 == 0:
            truth.append((sim, x, 0.0, 0.0))
        if index % 20 == 0:
            amcl.append((sim, x + 0.05, 0.0, 0.0))
    start_wall = wall_start + 2.0
    end_wall = wall_start + seconds
    first = {"/odom": wall_start, "/cmd_vel": start_wall + 0.1, "/amcl_pose": wall_start,
             "/ground_truth_pose": wall_start, "/scan": wall_start, "/plan": start_wall + 0.001,
             "/behavior_tree_log": start_wall}
    last = {topic: end_wall for topic in first}
    if commands_until is not None:
        # Commands stop shortly after the terminal outcome, as in the real recordings.
        last["/cmd_vel"] = min(end_wall, start_wall + commands_until)
        commands = [item for item in commands if item[0] <= last["/cmd_vel"]]
    if not bt:
        first.pop("/behavior_tree_log")
        last.pop("/behavior_tree_log")
    counts = {topic: 100 for topic in first}
    counts["/collision_event"] = 1 if contact_after_start is not None else 0
    if topics is not None:
        counts = {topic: counts.get(topic, 0) for topic in topics}
    return BagEvidence(
        topic_counts=counts, first_receive=first, last_receive=last, clock_knots=knots,
        first_contact_receive=(
            None if contact_after_start is None else start_wall + contact_after_start
        ),
        commands=commands, odometry=odometry, ground_truth=truth, amcl=amcl,
    )


def test_event_class_mapping_and_label_only_family_rule():
    assert TERMINAL_EVENT_CLASS == {
        "collision": "collision", "planner_failure": "navigation_abort",
        "timeout": "mission_timeout", "false_arrival": "false_arrival", "success": None,
    }
    assert research2_fault_family("clean") == "none"
    assert research2_fault_family("fog") == "research1_fog"
    assert research2_severity("clean", 0) == "none"
    assert research2_severity("fog", "3") == "research1_3"
    for family in RESEARCH2_FAULT_FAMILIES:
        assert research2_fault_family(family) == f"research1_{family}"
    with pytest.raises(ValueError):
        research2_fault_family("")
    assert episode_key("abc") == "r1-abc"


def test_clock_map_interpolates_and_reports_real_time_factor():
    clock = ReceiveClockMap([(0.0, 10.0), (1.0, 10.8), (2.0, 11.6)])
    assert clock.to_sim(0.5) == pytest.approx(10.4)
    assert clock.to_sim(3.0) == pytest.approx(12.4)  # extrapolates with the last slope
    assert clock.real_time_factor == pytest.approx(0.8)
    with pytest.raises(ValueError):
        ReceiveClockMap([(0.0, 1.0)])


def test_collision_is_exact_from_contact_and_wall_duration_is_converted():
    # duration_s in wall seconds (pre-v1.21 commit): 20 wall s * 0.8 = 16 sim s.
    contact = 19.9
    result = assess_episode(
        CATALOG_ROW, aggregate(duration_s=20.0), ROUTE,
        evidence(contact_after_start=contact, commands_until=20.3),
        event_config=EVENT_CONFIG, duration_time_base="wall_clock",
    )
    assert result["admissible"], result["reasons"]
    start = result["episode_start_sim"]
    assert start == pytest.approx(10.0 + 2.0 * 0.8)
    assert result["terminal_event_class"] == "collision"
    assert result["terminal_time_exactness"] == "exact"
    assert result["terminal_event_time_sim"] == pytest.approx(start + contact * 0.8)
    assert result["collision_residual_s"] == pytest.approx(-0.1 * 0.8)
    assert result["primary_event_class"] == "collision"
    assert result["fault_family"] == "research1_lighting"
    assert result["severity"] == "research1_2"
    assert "/local_plan" in result["missing_channels"]
    sidecar = build_event_sidecar(result)
    kinds = [item["event_type"] for item in sidecar]
    assert "goal_dispatched" in kinds and "terminal_event" in kinds
    terminal = next(item for item in sidecar if item["event_type"] == "terminal_event")
    assert terminal["parameters"]["terminal_state"] == "collision"


def test_planner_failure_is_bounded_and_simulation_duration_is_used_directly():
    result = assess_episode(
        CATALOG_ROW, aggregate(terminal_state="planner_failure", collision="False",
                               duration_s=30.0),
        ROUTE, evidence(rtf=0.8, seconds=40.0), event_config=EVENT_CONFIG,
        duration_time_base="simulation",
    )
    assert result["admissible"], result["reasons"]
    assert result["mapped_event_class"] == "navigation_abort"
    assert result["terminal_time_exactness"] == "bounded_upper"
    assert result["terminal_event_time_sim"] == pytest.approx(result["episode_start_sim"] + 30.0)


def test_duration_time_base_is_inferred_from_bag_when_commit_is_unknown():
    # 30 wall s -> 24 sim s: only the wall interpretation ends inside the recording.
    bag = evidence(rtf=0.8, seconds=32.0)
    result = assess_episode(
        CATALOG_ROW, aggregate(terminal_state="planner_failure", collision="False",
                               duration_s=30.0),
        ROUTE, bag, event_config=EVENT_CONFIG, duration_time_base=None,
    )
    assert result["admissible"], result["reasons"]
    assert result["aggregate_duration_time_base"] == "wall_clock"
    assert result["duration_time_base_source"] == "bag_evidence"


@pytest.mark.parametrize("overrides,bag_kwargs,expected", [
    ({"invalid_reason": "logging_failure", "terminal_state": "invalid"}, {}, "aggregate_invalid"),
    ({"terminal_state": "collision"}, {"contact_after_start": None},
     "collision_without_contact_evidence"),
    ({"terminal_state": "timeout", "collision": "False", "duration_s": 100.0},
     {"seconds": 140.0, "rtf": 1.0}, "timeout_not_at_route_budget"),
    ({"terminal_state": "false_arrival", "collision": "False", "goal_distance_gt_m": 0.1},
     {}, "false_arrival_within_goal_tolerance"),
    ({"terminal_state": "success", "collision": "False", "success": "True",
      "goal_distance_gt_m": 0.9}, {}, "success_outside_goal_tolerance"),
    ({"terminal_state": "collision", "duration_s": 3.0}, {"contact_after_start": 2.9},
     "episode_shorter_than_history_window"),
    ({"map_id": "dev_02"}, {"contact_after_start": 19.9}, "identity_mismatch:map_id"),
    ({"terminal_state": "planner_failure", "collision": "False", "duration_s": 200.0},
     {}, "terminal_beyond_bag_end"),
])
def test_rejection_reasons(overrides, bag_kwargs, expected):
    kwargs = {"contact_after_start": 19.9}
    kwargs.update(bag_kwargs)
    result = assess_episode(
        CATALOG_ROW, aggregate(**overrides), ROUTE, evidence(**kwargs),
        event_config=EVENT_CONFIG, duration_time_base="simulation",
    )
    assert not result["admissible"]
    assert expected in result["reasons"], result["reasons"]


def test_missing_required_topic_and_missing_route_are_rejected():
    bag = evidence(contact_after_start=19.9, topics=["/odom", "/cmd_vel", "/amcl_pose", "/plan"])
    result = assess_episode(CATALOG_ROW, aggregate(), None, bag, event_config=EVENT_CONFIG,
                            duration_time_base="simulation")
    assert not result["admissible"]
    assert "missing_required_topic:/scan" in result["reasons"]
    assert "route_spec_missing" in result["reasons"]
    duplicate = assess_episode(CATALOG_ROW, aggregate(), ROUTE, evidence(contact_after_start=19.9),
                               event_config=EVENT_CONFIG, duplicate_run_id=True,
                               duration_time_base="simulation")
    assert "duplicate_run_id" in duplicate["reasons"]


def test_supplement_sizing_arithmetic():
    assert supplement_episode_count(825) == 324      # 315 -> floor 324, already 27 blocks
    assert supplement_episode_count(821) == 324      # 319 -> 324
    assert supplement_episode_count(700) == 444      # 440 -> 37 blocks
    assert supplement_episode_count(0) == 1140
    assert supplement_episode_count(5000) == 324
    assert distribute_blocks(27, 9) == [3] * 9
    assert distribute_blocks(37, 9) == [5, 4, 4, 4, 4, 4, 4, 4, 4]
    with pytest.raises(ValueError):
        supplement_episode_count(-1)


def test_supplement_keys_and_seeds_never_collide_with_existing_campaigns():
    balanced = yaml.safe_load((ROOT / "data/manifests/balanced_pilot_v1.yaml").read_text())
    targeted = yaml.safe_load((ROOT / "data/manifests/targeted_development_v1.yaml").read_text())
    splits = yaml.safe_load((ROOT / "data/manifests/splits.template.yaml").read_text())
    for admitted in (825, 700, 0):
        document = build_supplement(
            admitted, balanced, targeted, audit_path="audit.yaml", audit_sha256="0" * 64,
            now_utc="2026-09-03T00:00:00Z",
        )
        assert collision_findings(document, balanced, targeted) == []
        expanded = expand_balanced_pilot(document)
        assert len(expanded) == document["expected_episode_count"] == supplement_episode_count(admitted)
        assert document["design"]["seed_base"] == 2_000_000
        assert min(item["seed"] for item in expanded) >= 2_000_000
        assert document["status"] == "preregistered_waiting_for_targeted_completion"
        assert document["allowed_splits"] == ["development"]
        assert document["protected_test_used"] is False
        assert document["execution_policy"]["exact_once_per_episode_key"] is True
        families = {item["family"] for item in expanded}
        assert families == RESEARCH2_FAULT_FAMILIES | {"none"}
        assert {item["system"] for item in expanded if item["family"] == "none"} == {"s0", "s3"}
        for item in expanded:
            assert not item["episode_key"].startswith("r1-")
    balanced_document = build_supplement(
        825, balanced, targeted, audit_path="a", audit_sha256="0" * 64, now_utc="t",
    )
    assert balanced_document["expected_per_condition"] == 36
    assert validate_balanced_pilot(balanced_document, splits) == []
    # Adapter keys use the Research 1 uuid namespace and cannot match the design grammar.
    all_keys = {item["episode_key"] for item in expand_balanced_pilot(balanced)}
    all_keys |= {item["episode_key"] for item in expand_balanced_pilot(targeted)}
    assert episode_key("0054e171-df5e-4470-9788-9ff00af04e09") not in all_keys


def test_native_research2_summaries_take_the_unchanged_pipeline_path(tmp_path):
    """The shared per-episode scripts only branch on fields adapted summaries carry."""
    from src.research1_adapter import load_receive_clock_map
    from scripts.derive_operational_events import read_evidence
    from scripts.extract_episode_annotation import read_sidecar_events
    import inspect

    native = yaml.safe_load(
        (ROOT / "data/raw/summaries/002e46af-5e2b-4700-8b29-99d56be21d67.yaml").read_text()
    )
    assert load_receive_clock_map(native) is None
    assert native["provenance"].get("event_source") is None
    assert native["provenance"].get("receive_clock_map") is None
    assert inspect.signature(read_evidence).parameters["receive_clock"].default is None
    # An adapted summary with a checksummed clock map round-trips through the loader.
    clock = ReceiveClockMap([(0.0, 10.0), (1.0, 10.8)])
    clock_path = tmp_path / "clock.json"
    clock_path.write_text(__import__("json").dumps(clock.to_json()))
    import hashlib
    adapted = {"provenance": {
        "time_base": "simulation_time_via_receive_clock_map",
        "receive_clock_map": str(clock_path),
        "receive_clock_map_sha256": hashlib.sha256(clock_path.read_bytes()).hexdigest(),
        "event_sidecar": str(tmp_path / "events.json"),
    }, "identity": {"run_id": "r"}}
    assert load_receive_clock_map(adapted).to_sim(0.5) == pytest.approx(10.4)
    (tmp_path / "events.json").write_text(__import__("json").dumps([
        {"event_type": "goal_dispatched", "parameters": {}, "reason": "",
         "stamp": {"sec": 12, "nanosec": 500000000}},
    ]))
    events = read_sidecar_events(adapted)
    assert events[0]["time"] == pytest.approx(12.5) and events[0]["run_id"] == "r"


def test_supplement_refuses_key_collision_when_replicates_overlap():
    balanced = yaml.safe_load((ROOT / "data/manifests/balanced_pilot_v1.yaml").read_text())
    targeted = yaml.safe_load((ROOT / "data/manifests/targeted_development_v1.yaml").read_text())
    document = build_supplement(825, balanced, targeted, audit_path="a", audit_sha256="0" * 64,
                                now_utc="t")
    clashing = copy.deepcopy(document)
    clashing["design"]["conditions"][2]["replicates"] = [10, 11, 12]  # targeted camera range
    findings = collision_findings(clashing, balanced, targeted)
    assert any("targeted_development_v1" in finding for finding in findings)

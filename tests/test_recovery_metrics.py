import pytest

from src.evaluation import evaluate_paired_recovery


def row(policy, seed, complete, collision=False, **changes):
    value = {
        "policy_id": policy, "map_id": "dev_00", "route_id": "dev_00_r0",
        "seed": seed, "fault_family": "lidar_dropout", "severity": "medium",
        "mission_complete": complete, "collision": collision,
        "guard_violation": False, "guard_rejected": False,
        "added_time_seconds": 2.0, "added_path_length_m": 0.5,
        "intervention_count": 1, "recovery_action": "controlled_stop",
    }
    value.update(changes)
    return value


def test_recovery_metrics_preserve_pairs_and_report_safety_and_overhead():
    records = [
        row("R0", 1, False), row("R3", 1, True),
        row("R0", 2, True), row("R3", 2, True, guard_rejected=True),
    ]
    report = evaluate_paired_recovery(records, "R0", "R3")
    assert report["pair_count"] == 2
    assert report["paired_completion_rate_difference"] == 0.5
    assert report["paired_collision_rate_difference"] == 0.0
    assert report["proposed"]["guard_rejection_count"] == 1
    assert report["safety_gate_passed"] is True


def test_recovery_metrics_reject_unmatched_or_duplicate_pairs():
    with pytest.raises(ValueError, match="unmatched"):
        evaluate_paired_recovery([row("R0", 1, False)], "R0", "R3")
    duplicate = [row("R0", 1, False), row("R0", 1, False), row("R3", 1, True)]
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_paired_recovery(duplicate, "R0", "R3")


def test_recovery_metrics_guard_violation_fails_safety_gate():
    report = evaluate_paired_recovery(
        [row("R0", 1, False), row("R3", 1, True, guard_violation=True)], "R0", "R3"
    )
    assert report["safety_gate_passed"] is False

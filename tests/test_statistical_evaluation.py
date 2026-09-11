import pytest

from src.evaluation import (
    brier_score,
    hierarchical_paired_binary_bootstrap,
    reliability_curve,
    select_validation_threshold,
)


def test_calibration_uses_only_eligible_decisions_and_handles_probability_one():
    rows = [
        {"risk_score": 0.0, "eligibility": "eligible_negative"},
        {"risk_score": 1.0, "eligibility": "eligible_positive"},
        {"risk_score": 0.9, "eligibility": "excluded_too_late"},
    ]
    assert brier_score(rows) == 0.0
    report = reliability_curve(rows, bins=2)
    assert report["eligible_decision_count"] == 2
    assert report["ece"] == 0.0
    assert report["bins"][1]["count"] == 1


def test_calibration_rejects_invalid_probabilities_and_empty_eligible_set():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([{"risk_score": 1.1, "eligibility": "eligible_positive"}])
    with pytest.raises(ValueError, match="no eligible"):
        reliability_curve([{"risk_score": 0.5, "eligibility": "excluded_too_late"}])


def test_hierarchical_bootstrap_is_paired_deterministic_and_episode_level():
    records = [
        {"map_id": "m0", "route_id": "r0", "episode_id": "e0", "event": True,
         "tcn_detected": True, "rules_detected": False},
        {"map_id": "m0", "route_id": "r0", "episode_id": "e1", "event": True,
         "tcn_detected": True, "rules_detected": True},
        {"map_id": "m1", "route_id": "r1", "episode_id": "e2", "event": True,
         "tcn_detected": False, "rules_detected": False},
        {"map_id": "m1", "route_id": "r1", "episode_id": "clean", "event": False,
         "tcn_detected": True, "rules_detected": False},
    ]
    report = hierarchical_paired_binary_bootstrap(
        records, "tcn_detected", "rules_detected", include_field="event",
        replicates=200, seed=7,
    )
    assert report["episode_count"] == 3
    assert report["map_count"] == 2
    assert report["point_estimate"] == pytest.approx(1 / 3)
    assert report == hierarchical_paired_binary_bootstrap(
        records, "tcn_detected", "rules_detected", include_field="event",
        replicates=200, seed=7,
    )


def test_hierarchical_bootstrap_rejects_duplicate_episode_identity():
    row = {"map_id": "m", "route_id": "r", "episode_id": "e", "a": True, "b": False}
    with pytest.raises(ValueError, match="unique"):
        hierarchical_paired_binary_bootstrap([row, row], "a", "b")


def test_validation_threshold_obeys_clean_mission_budget_and_is_deterministic():
    episodes = {
        "clean": [
            {"decision_time": 1.0, "risk_score": 0.4, "eligibility": "eligible_negative",
             "primary_event_time": None},
            {"decision_time": 2.0, "risk_score": 0.4, "eligibility": "eligible_negative",
             "primary_event_time": None},
        ],
        "event": [
            {"decision_time": 1.0, "risk_score": 0.8, "eligibility": "eligible_positive",
             "primary_event_time": 5.0},
            {"decision_time": 2.0, "risk_score": 0.8, "eligibility": "eligible_positive",
             "primary_event_time": 5.0},
        ],
    }
    selected = select_validation_threshold(
        episodes, {"clean"}, false_alert_budget=0.0,
        required_above=1, decisions_considered=1, cooldown_seconds=10.0,
    )
    assert selected["threshold"] == 0.8
    assert selected["event_recall"] == 1.0
    assert selected["false_alerts_per_clean_mission"] == 0.0


def test_validation_threshold_requires_known_clean_missions():
    with pytest.raises(ValueError, match="clean validation"):
        select_validation_threshold({}, set(), false_alert_budget=0.1)

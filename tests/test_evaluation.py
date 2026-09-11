from src.evaluation import AlarmPolicy, apply_alarm_policy, evaluate_event_warnings
from src.models import Rule, ThresholdRuleSet


def test_threshold_rules_respect_missingness_and_command_gate():
    rules = ThresholdRuleSet([
        Rule("stuck", "progress", "below", 0.1, "command", "above", 0.05),
        Rule("dropout", "valid", "below", 0.5),
    ])
    assert rules.predict({"progress": 0.0, "command": 0.1, "valid": 1.0})["fired_rules"] == ["stuck"]
    assert rules.predict({"progress": 0.0, "command": 0.0, "valid": 1.0})["risk_score"] == 0.0
    assert rules.predict({"progress": 0.0, "command": 0.1, "valid": 0.0,
                          "valid__missing": 1})["fired_rules"] == ["stuck"]


def test_policy_applies_two_of_three_and_cooldown():
    rows = [{"decision_time": float(i), "risk_score": score}
            for i, score in enumerate([1, 0, 1, 1, 1, 1])]
    output = apply_alarm_policy(rows, AlarmPolicy(0.5, 2, 3, 3.0))
    assert [row["alarm"] for row in output] == [False, False, True, False, False, True]


def test_event_metrics_credit_only_useful_warning_and_retain_undetected_denominator():
    episodes = {
        "detected": [
            {"decision_time": 10, "alarm": True, "eligibility": "eligible_negative",
             "primary_event_time": 30},
            {"decision_time": 22, "alarm": True, "eligibility": "eligible_positive",
             "primary_event_time": 30},
        ],
        "missed": [
            {"decision_time": 22, "alarm": False, "eligibility": "eligible_positive",
             "primary_event_time": 30},
        ],
        "clean": [
            {"decision_time": 10, "alarm": True, "eligibility": "eligible_negative",
             "primary_event_time": None},
        ],
    }
    metrics = evaluate_event_warnings(episodes)
    assert metrics["event_recall"] == 0.5
    assert metrics["undetected_event_count"] == 1
    assert metrics["median_useful_lead_seconds_detected"] == 8.0
    assert metrics["false_alert_count"] == 2
    assert metrics["false_alerts_per_non_event_mission"] == 1.0


def test_threshold_candidates_are_capped_for_dense_scores():
    from src.evaluation import select_validation_threshold
    episodes = {}
    for episode in range(4):
        rows = []
        for index in range(600):
            score = ((episode * 600 + index) % 997) / 997.0
            rows.append({
                "decision_time": 5.0 + 0.5 * index, "risk_score": score,
                "eligibility": "eligible_negative", "primary_event_time": None,
            })
        episodes[f"clean-{episode}"] = rows
    report = select_validation_threshold(
        episodes, set(episodes), false_alert_budget=0.1, maximum_candidates=51,
    )
    assert report["distinct_validation_scores"] > 51
    assert report["candidate_threshold_count"] <= 51
    assert report["false_alerts_per_clean_mission"] <= 0.1

import numpy as np
import pytest

from src.evaluation import AlarmPolicy, apply_alarm_policy
from src.evaluation.discrimination import (
    alert_burden_summary, auprc, auroc, discrimination_from_rows,
    hierarchical_episode_bootstrap, lead_time_summary, paired_event_recall_difference,
)
from src.evaluation.event_metrics import evaluate_event_warnings
from src.evaluation.prediction_tables import group_episodes
from synthetic_prediction_tables import build_rows


def test_auroc_and_auprc_hand_values_with_ties():
    scores = [0.9, 0.8, 0.8, 0.1]
    labels = [1, 0, 1, 0]
    # ranks: 0.1 -> 1, 0.8 -> 2.5, 2.5, 0.9 -> 4; positives ranks 4 + 2.5 = 6.5
    assert auroc(scores, labels) == pytest.approx((6.5 - 3) / 4)
    # thresholds: 0.9 -> P=1 R=.5 ; 0.8 -> P=2/3 R=1 ; AP = .5*1 + .5*2/3
    assert auprc(scores, labels) == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert auroc([0.2, 0.3], [1, 1]) is None
    assert auprc([0.2, 0.3], [0, 0]) is None


def test_discrimination_matches_sklearn_when_available():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    scores = np.round(rng.random(300), 2)
    labels = (rng.random(300) < scores).astype(int)
    assert auroc(scores, labels) == pytest.approx(sklearn.roc_auc_score(labels, scores))
    assert auprc(scores, labels) == pytest.approx(sklearn.average_precision_score(labels, scores))


def test_discrimination_uses_only_eligible_rows():
    rows = build_rows("p3_causal_tcn")
    report = discrimination_from_rows(rows)
    assert report["eligible_positive_count"] == 4 * 9
    assert report["eligible_negative_count"] == 4 * 20 + 4 * 10 + 20
    assert report["role"] == "supporting_metric_only"
    assert 0.0 < report["auprc"] < 1.0


def test_hierarchical_episode_bootstrap_is_deterministic_and_episode_grouped():
    rows = build_rows("p3_causal_tcn")

    def episodes_seen(sample):
        return float(len({row["run_id"] for row in sample}))

    report = hierarchical_episode_bootstrap(rows, episodes_seen, replicates=50, seed=1)
    assert report["episode_count"] == 9
    assert report["map_count"] == 2
    assert report["point_estimate"] == 9.0
    assert report["confidence_interval"][0] <= 9.0 <= report["confidence_interval"][1] + 9
    assert report == hierarchical_episode_bootstrap(rows, episodes_seen, replicates=50, seed=1)
    undefined = hierarchical_episode_bootstrap(rows, lambda sample: None, replicates=5, seed=1)
    assert undefined["undefined_replicates"] == 5
    assert undefined["confidence_interval"] is None


def _alarmed(rows, threshold):
    policy = AlarmPolicy(threshold, 2, 3, 10.0)
    return [row for _run, episode in group_episodes(rows).items() for row in apply_alarm_policy(episode, policy)]


def test_paired_recall_difference_matches_hand_calculation():
    p3 = _alarmed(build_rows("p3_causal_tcn"), 0.9)
    p1 = _alarmed(build_rows("p1_threshold_rules"), 1.0)
    report = paired_event_recall_difference(p3, p1, replicates=100, seed=5)
    assert report["event_recall_a"] == 0.75
    assert report["event_recall_b"] == 0.25
    assert report["point_estimate"] == pytest.approx(0.5)
    assert report["event_episode_count"] == 4
    assert report["total_episode_count"] == 9
    assert report["hierarchy"] == ["map", "route", "episode"]
    with pytest.raises(ValueError, match="identical episode sets"):
        paired_event_recall_difference(p3, [row for row in p1 if row["run_id"] != "E1"])


def test_lead_time_and_burden_summaries_keep_full_denominator():
    p3 = _alarmed(build_rows("p3_causal_tcn"), 0.9)
    metrics = evaluate_event_warnings(group_episodes(p3))
    leads = lead_time_summary(metrics)
    assert leads["event_count"] == 4 and leads["undetected_event_count"] == 1
    assert leads["median_seconds_detected"] == 4.5
    assert leads["fraction_of_all_events_warned_at_least"]["3s"] == 0.75
    assert leads["fraction_of_all_events_warned_at_least"]["5s"] == 0.0
    burden = alert_burden_summary(p3, decision_rate_hz=2.0)
    assert burden["clean"]["total_alerts"] == 0
    assert burden["faulted"]["total_alerts"] == 3
    assert burden["faulted"]["repeated_alerts_per_mission"] == 0.0
    # persistent from t=5.5 to 9.5 -> 9 decisions = 4.5 s per detected episode
    assert burden["faulted"]["time_under_alert_seconds_per_mission"] == pytest.approx(3 * 4.5 / 5)

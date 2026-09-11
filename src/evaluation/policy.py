"""Threshold, persistence, and cooldown logic for warning scores."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

from .event_metrics import evaluate_event_warnings


@dataclass(frozen=True)
class AlarmPolicy:
    threshold: float
    required_above: int = 2
    decisions_considered: int = 3
    cooldown_seconds: float = 10.0

    def validate(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must lie in [0, 1]")
        if not 1 <= self.required_above <= self.decisions_considered:
            raise ValueError("invalid M-of-N persistence")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown must be nonnegative")


def apply_alarm_policy(
    rows: Sequence[Mapping[str, object]], policy: AlarmPolicy
) -> list[dict[str, object]]:
    policy.validate()
    history: deque[bool] = deque(maxlen=policy.decisions_considered)
    last_alarm = float("-inf")
    output = []
    previous_time = float("-inf")
    for source in rows:
        decision_time = float(source["decision_time"])
        if decision_time <= previous_time:
            raise ValueError("policy rows must be strictly time ordered within an episode")
        score = float(source["risk_score"])
        history.append(score >= policy.threshold)
        persistent = len(history) == policy.decisions_considered and sum(history) >= policy.required_above
        alarm = persistent and decision_time - last_alarm >= policy.cooldown_seconds
        if alarm:
            last_alarm = decision_time
        output.append({**source, "alarm": alarm, "persistent": persistent})
        previous_time = decision_time
    return output


def select_validation_threshold(
    episodes: Mapping[str, Sequence[Mapping[str, object]]],
    clean_run_ids: set[str],
    *,
    false_alert_budget: float,
    required_above: int = 2,
    decisions_considered: int = 3,
    cooldown_seconds: float = 10.0,
    maximum_candidates: int = 1001,
) -> dict[str, object]:
    """Maximize event recall subject to the frozen clean-mission alarm budget.

    Every distinct validation score is a candidate threshold while there are at most
    ``maximum_candidates`` of them; learned predictors emit tens of thousands of
    distinct scores, so above that count the candidates are the score quantiles at
    ``maximum_candidates`` evenly spaced probabilities plus 0 and 1. The search stays
    deterministic and validation-only either way.
    """
    if not clean_run_ids:
        raise ValueError("threshold selection requires clean validation missions")
    unknown = clean_run_ids - set(episodes)
    if unknown:
        raise ValueError(f"clean run ids absent from validation episodes: {sorted(unknown)}")
    if false_alert_budget < 0:
        raise ValueError("false-alert budget must be nonnegative")
    scores = {
        float(row["risk_score"])
        for rows in episodes.values() for row in rows
    }
    if not scores or any(not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("validation risk scores must lie in [0, 1]")
    if maximum_candidates < 2:
        raise ValueError("maximum_candidates must be at least 2")
    if len(scores) + 2 <= maximum_candidates:
        candidates = sorted({0.0, 1.0, *scores})
    else:
        ordered = sorted(scores)
        interior = max(maximum_candidates - 2, 1)
        positions = [
            round(index * (len(ordered) - 1) / max(interior - 1, 1))
            for index in range(interior)
        ]
        candidates = sorted({0.0, 1.0, *(ordered[position] for position in positions)})
    feasible = []
    for threshold in candidates:
        policy = AlarmPolicy(
            threshold, required_above, decisions_considered, cooldown_seconds
        )
        predicted = {
            run_id: apply_alarm_policy(rows, policy)
            for run_id, rows in episodes.items()
        }
        metrics = evaluate_event_warnings(predicted)
        clean_false_alerts = sum(
            episode["false_alerts"] for episode in metrics["per_episode"]
            if episode["run_id"] in clean_run_ids
        )
        burden = clean_false_alerts / len(clean_run_ids)
        if burden <= false_alert_budget:
            recall = metrics["event_recall"]
            lead = metrics["median_useful_lead_seconds_detected"]
            feasible.append({
                "threshold": threshold,
                "false_alerts_per_clean_mission": burden,
                "event_recall": recall,
                "median_useful_lead_seconds_detected": lead,
                "metrics": metrics,
            })
    if not feasible:
        raise ValueError("no threshold satisfies the clean-mission false-alert budget")

    def ranking(item: Mapping[str, object]) -> tuple[float, float, float, float]:
        recall = item["event_recall"]
        lead = item["median_useful_lead_seconds_detected"]
        return (
            float(recall) if recall is not None else -1.0,
            -float(item["false_alerts_per_clean_mission"]),
            float(lead) if lead is not None else -1.0,
            float(item["threshold"]),
        )

    selected = max(feasible, key=ranking)
    return {
        "selection_split": "validation",
        "objective": "maximum_event_recall_subject_to_false_alert_budget",
        "false_alert_budget_per_clean_mission": false_alert_budget,
        "candidate_threshold_count": len(candidates),
        "distinct_validation_scores": len(scores),
        "maximum_candidates": maximum_candidates,
        "feasible_threshold_count": len(feasible),
        **selected,
    }

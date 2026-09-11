"""Window-level discrimination (supporting metrics) with episode-grouped bootstraps.

Windows are never treated as independent for uncertainty: every interval here
resamples maps, then routes, then episodes, mirroring ``bootstrap.py``.
"""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Callable, Mapping, Sequence

import numpy as np

from .bootstrap import _percentile, hierarchical_paired_binary_bootstrap
from .event_metrics import evaluate_event_warnings
from .prediction_tables import ELIGIBLE, group_episodes


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Mann-Whitney AUROC with average ranks for ties; None without both classes."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=float)
    index = 0
    while index < scores.size:
        end = index
        while end + 1 < scores.size and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        ranks[order[index:end + 1]] = (index + end) / 2.0 + 1.0
        index = end + 1
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def auprc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Average precision over distinct descending thresholds (step-wise, no interpolation)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    positives = int(labels.sum())
    if positives == 0 or labels.size == positives:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels)
    predicted_positive = np.arange(1, labels.size + 1)
    distinct = np.flatnonzero(np.diff(sorted_scores)) if labels.size > 1 else np.array([], int)
    boundaries = np.append(distinct, labels.size - 1)
    precision = true_positive[boundaries] / predicted_positive[boundaries]
    recall = true_positive[boundaries] / positives
    previous_recall = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - previous_recall) * precision))


def eligible_arrays(rows: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    scores, labels = [], []
    for row in rows:
        if row.get("eligibility") not in ELIGIBLE:
            continue
        scores.append(float(row["risk_score"]))
        labels.append(1 if row.get("eligibility") == "eligible_positive" else 0)
    return np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)


def discrimination_from_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    scores, labels = eligible_arrays(rows)
    return {
        "auprc": auprc(scores, labels),
        "auroc": auroc(scores, labels),
        "eligible_positive_count": int(labels.sum()),
        "eligible_negative_count": int(labels.size - labels.sum()),
        "unit": "eligible_decision_window",
        "role": "supporting_metric_only",
    }


def _hierarchy(rows: Sequence[Mapping[str, object]]):
    hierarchy: dict[str, dict[str, dict[str, list[Mapping[str, object]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        hierarchy[str(row["map_id"])][str(row["route_id"])][str(row["run_id"])].append(row)
    return hierarchy


def hierarchical_episode_bootstrap(
    rows: Sequence[Mapping[str, object]],
    metric: Callable[[Sequence[Mapping[str, object]]], float | None],
    *,
    replicates: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260903,
) -> dict[str, object]:
    """Map -> route -> episode resampling of a scalar metric of a prediction table."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if not rows:
        raise ValueError("no rows to bootstrap")
    hierarchy = _hierarchy(rows)
    point = metric(rows)
    maps = sorted(hierarchy)
    rng = random.Random(seed)
    draws: list[float] = []
    undefined = 0
    for _ in range(replicates):
        sampled: list[Mapping[str, object]] = []
        for _map_draw in range(len(maps)):
            map_id = rng.choice(maps)
            routes = sorted(hierarchy[map_id])
            for _route_draw in range(len(routes)):
                route_id = rng.choice(routes)
                episodes = sorted(hierarchy[map_id][route_id])
                for _episode_draw in range(len(episodes)):
                    sampled.extend(hierarchy[map_id][route_id][rng.choice(episodes)])
        value = metric(sampled)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            undefined += 1
        else:
            draws.append(float(value))
    alpha = (1.0 - confidence) / 2.0
    interval = [_percentile(draws, alpha), _percentile(draws, 1.0 - alpha)] if draws else None
    episode_count = sum(len(routes[route]) for routes in hierarchy.values() for route in routes)
    return {
        "analysis_unit": "episode",
        "hierarchy": ["map", "route", "episode"],
        "episode_count": episode_count,
        "map_count": len(maps),
        "point_estimate": point,
        "confidence": confidence,
        "confidence_interval": interval,
        "bootstrap_replicates": replicates,
        "undefined_replicates": undefined,
        "seed": seed,
    }


def _auprc_metric(rows: Sequence[Mapping[str, object]]) -> float | None:
    return auprc(*eligible_arrays(rows))


def _auroc_metric(rows: Sequence[Mapping[str, object]]) -> float | None:
    return auroc(*eligible_arrays(rows))


def discrimination_with_intervals(
    rows: Sequence[Mapping[str, object]], *, replicates: int = 1000, seed: int = 20260903
) -> dict[str, object]:
    report = discrimination_from_rows(rows)
    report["auprc_bootstrap"] = hierarchical_episode_bootstrap(
        rows, _auprc_metric, replicates=replicates, seed=seed
    )
    report["auroc_bootstrap"] = hierarchical_episode_bootstrap(
        rows, _auroc_metric, replicates=replicates, seed=seed
    )
    return report


def episode_detection_records(
    rows: Sequence[Mapping[str, object]]
) -> dict[str, dict[str, object]]:
    """Per-episode detection flags from a table that already carries ``alarm``."""
    episodes = group_episodes(rows)
    metrics = evaluate_event_warnings(episodes)
    identity = {
        run_id: (str(episode_rows[0]["map_id"]), str(episode_rows[0]["route_id"]))
        for run_id, episode_rows in episodes.items()
    }
    records = {}
    for episode in metrics["per_episode"]:
        map_id, route_id = identity[episode["run_id"]]
        records[episode["run_id"]] = {
            "map_id": map_id, "route_id": route_id, "episode_id": episode["run_id"],
            "event": episode["has_event"], "detected": episode["detected"],
            "false_alerts": episode["false_alerts"],
        }
    return records


def paired_event_recall_difference(
    rows_a: Sequence[Mapping[str, object]],
    rows_b: Sequence[Mapping[str, object]],
    *,
    replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260903,
) -> dict[str, object]:
    """H1 estimator: A-minus-B event recall over the same episodes, hierarchical bootstrap."""
    records_a = episode_detection_records(rows_a)
    records_b = episode_detection_records(rows_b)
    if set(records_a) != set(records_b):
        raise ValueError("paired comparison requires identical episode sets")
    paired = []
    for run_id, record in sorted(records_a.items()):
        other = records_b[run_id]
        if record["event"] != other["event"]:
            raise ValueError(f"{run_id}: event status differs between tables")
        paired.append({
            "map_id": record["map_id"], "route_id": record["route_id"], "episode_id": run_id,
            "event": record["event"], "a_detected": record["detected"],
            "b_detected": other["detected"],
        })
    events = [record for record in paired if record["event"]]
    if not events:
        raise ValueError("paired comparison requires at least one event episode")
    report = hierarchical_paired_binary_bootstrap(
        paired, "a_detected", "b_detected", include_field="event",
        replicates=replicates, confidence=confidence, seed=seed,
    )
    report["event_recall_a"] = sum(record["a_detected"] for record in events) / len(events)
    report["event_recall_b"] = sum(record["b_detected"] for record in events) / len(events)
    report["event_episode_count"] = len(events)
    report["total_episode_count"] = len(paired)
    return report


# --- event-level summaries shared by the validation-only reports -------------------


def _quantile(values: Sequence[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    return _percentile(ordered, probability) if ordered else None


def lead_time_summary(metrics: Mapping[str, object]) -> dict[str, object]:
    """Lead-time distribution with the full event denominator (undetected reported)."""
    leads = [
        float(episode["first_useful_lead_seconds"]) for episode in metrics["per_episode"]
        if episode["detected"] and episode["first_useful_lead_seconds"] is not None
    ]
    event_count = int(metrics["event_count"])
    survival = {}
    for horizon in (1.0, 2.0, 3.0, 5.0, 10.0):
        warned = sum(1 for lead in leads if lead >= horizon)
        survival[f"{horizon:g}s"] = warned / event_count if event_count else None
    return {
        "event_count": event_count,
        "detected_event_count": len(leads),
        "undetected_event_count": event_count - len(leads),
        "median_seconds_detected": _quantile(leads, 0.5),
        "q1_seconds_detected": _quantile(leads, 0.25),
        "q3_seconds_detected": _quantile(leads, 0.75),
        "min_seconds_detected": min(leads) if leads else None,
        "max_seconds_detected": max(leads) if leads else None,
        "fraction_of_all_events_warned_at_least": survival,
    }


def alert_burden_summary(
    rows: Sequence[Mapping[str, object]], *, decision_rate_hz: float
) -> dict[str, object]:
    """Total alerts, time under alert and repeated alerts, split clean versus faulted."""
    from .prediction_tables import truth  # local import avoids a cycle at module load

    episodes = group_episodes(rows)
    groups = {"all": [], "clean": [], "faulted": []}
    for run_id, episode_rows in episodes.items():
        alerts = sum(1 for row in episode_rows if truth(row.get("alarm")))
        persistent = sum(1 for row in episode_rows if truth(row.get("persistent")))
        record = {
            "alerts": alerts, "repeated": max(alerts - 1, 0),
            "under_alert_seconds": persistent / decision_rate_hz, "decisions": len(episode_rows),
        }
        groups["all"].append(record)
        groups["clean" if episode_rows[0].get("fault_family") == "none" else "faulted"].append(record)
    summary = {}
    for name, records in groups.items():
        count = len(records)
        summary[name] = {
            "mission_count": count,
            "total_alerts": sum(record["alerts"] for record in records),
            "alerts_per_mission": (
                sum(record["alerts"] for record in records) / count if count else None
            ),
            "repeated_alerts_per_mission": (
                sum(record["repeated"] for record in records) / count if count else None
            ),
            "time_under_alert_seconds_per_mission": (
                sum(record["under_alert_seconds"] for record in records) / count if count else None
            ),
            "fraction_of_decisions_under_alert": (
                sum(record["under_alert_seconds"] for record in records) * decision_rate_hz
                / sum(record["decisions"] for record in records)
                if records else None
            ),
        }
    return summary


def grouped_event_summary(
    rows: Sequence[Mapping[str, object]], field: str
) -> dict[str, dict[str, object]]:
    """Event recall and false alerts per mission by an episode-level field."""
    buckets: dict[str, dict[str, list[Mapping[str, object]]]] = defaultdict(dict)
    for run_id, episode_rows in group_episodes(rows).items():
        buckets[str(episode_rows[0].get(field))][run_id] = episode_rows
    output = {}
    for key in sorted(buckets):
        metrics = evaluate_event_warnings(buckets[key])
        output[key] = {
            "episode_count": metrics["episode_count"],
            "event_count": metrics["event_count"],
            "detected_event_count": metrics["detected_event_count"],
            "event_recall": metrics["event_recall"],
            "false_alerts_per_mission": metrics["false_alerts_per_mission"],
            "median_useful_lead_seconds_detected": metrics["median_useful_lead_seconds_detected"],
        }
    return output

"""Validation-only probability calibration diagnostics for causal decision rows."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def _eligible(rows: Sequence[Mapping[str, object]]) -> list[tuple[float, int]]:
    values = []
    for row in rows:
        if row.get("eligibility") not in {"eligible_positive", "eligible_negative"}:
            continue
        probability = float(row["risk_score"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("risk_score must be finite and lie in [0, 1]")
        target = 1 if row.get("eligibility") == "eligible_positive" else 0
        values.append((probability, target))
    if not values:
        raise ValueError("no eligible calibration rows")
    return values


def brier_score(rows: Sequence[Mapping[str, object]]) -> float:
    values = _eligible(rows)
    return sum((probability - target) ** 2 for probability, target in values) / len(values)


def reliability_curve(
    rows: Sequence[Mapping[str, object]], bins: int = 10
) -> dict[str, object]:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    values = _eligible(rows)
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probability, target in values:
        index = min(int(probability * bins), bins - 1)
        buckets[index].append((probability, target))
    curve = []
    weighted_gap = 0.0
    for index, bucket in enumerate(buckets):
        count = len(bucket)
        predicted = sum(value[0] for value in bucket) / count if count else None
        observed = sum(value[1] for value in bucket) / count if count else None
        gap = abs(predicted - observed) if count else None
        if gap is not None:
            weighted_gap += count * gap
        curve.append({
            "bin_index": index,
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "count": count,
            "mean_predicted_risk": predicted,
            "observed_event_frequency": observed,
            "absolute_gap": gap,
        })
    return {
        "eligible_decision_count": len(values),
        "bin_count": bins,
        "ece": weighted_gap / len(values),
        "brier_score": brier_score(rows),
        "bins": curve,
    }

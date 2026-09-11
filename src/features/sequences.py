"""Assemble complete, causal fixed-rate histories for temporal predictors."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
from typing import Mapping, Sequence

from .causal import FeatureSpec, ScalarSample, extract_decision_rows
from .leakage import LeakagePolicy
from .window_features import derive_window_features


@dataclass(frozen=True)
class SequenceExample:
    run_id: str
    decision_index: int
    decision_time: float
    label: int
    feature_names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]


def _grid(start: float, end: float, stride: float) -> list[float]:
    if not all(math.isfinite(value) for value in (start, end, stride)):
        raise ValueError("timeline bounds and stride must be finite")
    if stride <= 0 or end < start:
        raise ValueError("timeline requires positive stride and end >= start")
    current = Decimal(str(start)) + Decimal(str(stride))
    final = Decimal(str(end))
    step = Decimal(str(stride))
    result = []
    while current <= final:
        result.append(float(current))
        current += step
    return result


def model_columns(primary_features: Sequence[str]) -> tuple[str, ...]:
    if not primary_features or len(primary_features) != len(set(primary_features)):
        raise ValueError("primary features must be non-empty and unique")
    return tuple(
        column
        for name in primary_features
        for column in (name, f"{name}__age_seconds", f"{name}__missing")
    )


def assemble_causal_sequences(
    *,
    run_id: str,
    episode_start: float,
    episode_end: float,
    labels: Sequence[Mapping[str, object]],
    specs: Sequence[FeatureSpec],
    samples_by_feature: Mapping[str, Sequence[ScalarSample]],
    leakage_policy: LeakagePolicy,
    primary_features: Sequence[str],
    goal_x: float,
    goal_y: float,
    history_seconds: float = 5.0,
    stride_seconds: float = 0.5,
    include_ineligible: bool = False,
) -> list[SequenceExample]:
    """Return eligible examples with exactly one complete past-only history each.

    With ``include_ineligible`` every label-grid decision is returned and excluded
    decisions carry ``label=-1``; the feature history is built identically, so a
    deployed alarm policy can be replayed over the complete consecutive stream.
    """
    ratio = history_seconds / stride_seconds
    steps = round(ratio)
    if history_seconds <= 0 or stride_seconds <= 0 or not math.isclose(ratio, steps):
        raise ValueError("history must be a positive integer multiple of stride")
    sample_times = _grid(episode_start, episode_end, stride_seconds)
    raw_rows = extract_decision_rows(
        run_id=run_id,
        decision_times=sample_times,
        specs=specs,
        samples_by_feature=samples_by_feature,
        leakage_policy=leakage_policy,
    )
    for row in raw_rows:
        now = float(row["decision_time"])
        for name, value in row.items():
            if name.startswith("__audit_source_time__") and value is not None \
                    and float(value) > now:
                raise AssertionError(f"future source timestamp in {name}")
    enriched = derive_window_features(
        raw_rows, goal_x=goal_x, goal_y=goal_y, history_seconds=history_seconds
    )
    columns = model_columns(primary_features)
    available = set(enriched[0]) if enriched else set()
    missing_columns = sorted(set(columns) - available)
    if missing_columns:
        raise ValueError(f"primary model columns are absent: {missing_columns}")
    index_by_time = {Decimal(str(row["decision_time"])): index for index, row in enumerate(enriched)}
    examples = []
    previous_label_time = float("-inf")
    for label_row in labels:
        decision_time = float(label_row["decision_time"])
        if decision_time <= previous_label_time:
            raise ValueError("label decisions must be strictly ordered")
        previous_label_time = decision_time
        raw_label = label_row.get("label")
        if raw_label not in (0, 1, "0", "1"):
            if not include_ineligible:
                continue
            raw_label = -1
        end_index = index_by_time.get(Decimal(str(decision_time)))
        if end_index is None:
            raise ValueError(f"label decision is absent from feature grid: {decision_time}")
        start_index = end_index - steps + 1
        if start_index < 0:
            raise ValueError(f"eligible decision lacks a complete history: {decision_time}")
        window = enriched[start_index:end_index + 1]
        lower_bound = decision_time - history_seconds
        if len(window) != steps or not all(
            lower_bound < float(row["decision_time"]) <= decision_time for row in window
        ):
            raise AssertionError("sequence slice violates its causal history interval")
        values = tuple(tuple(float(row[column]) for column in columns) for row in window)
        examples.append(SequenceExample(
            run_id=run_id,
            decision_index=int(label_row["decision_index"]),
            decision_time=decision_time,
            label=int(raw_label),
            feature_names=columns,
            values=values,
        ))
    return examples

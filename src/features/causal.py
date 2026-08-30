"""Causal resampling of already-derived scalar telemetry channels."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .leakage import LeakagePolicy


@dataclass(frozen=True)
class ScalarSample:
    timestamp: float
    value: float


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: str
    maximum_age_seconds: float

    def validate(self) -> None:
        if not self.name or not self.source:
            raise ValueError("feature name and source are required")
        if self.maximum_age_seconds <= 0:
            raise ValueError(f"{self.name}: maximum age must be positive")


def _validated_samples(name: str, samples: Sequence[ScalarSample]) -> tuple[list[float], list[float]]:
    timestamps: list[float] = []
    values: list[float] = []
    previous = float("-inf")
    for sample in samples:
        timestamp = float(sample.timestamp)
        value = float(sample.value)
        if not math.isfinite(timestamp):
            raise ValueError(f"{name}: non-finite sample timestamp")
        if timestamp < previous:
            raise ValueError(f"{name}: samples are not monotonic")
        if not math.isfinite(value):
            raise ValueError(f"{name}: non-finite observed value")
        timestamps.append(timestamp)
        values.append(value)
        previous = timestamp
    return timestamps, values


def extract_decision_rows(
    *,
    run_id: str,
    decision_times: Sequence[float],
    specs: Sequence[FeatureSpec],
    samples_by_feature: Mapping[str, Sequence[ScalarSample]],
    leakage_policy: LeakagePolicy,
) -> list[dict[str, object]]:
    """Select the newest non-future sample and emit value, age, and missing channels.

    Missing or expired values are encoded as value=0, age=max_age, missing=1. The mask
    distinguishes that neutral storage value from a real zero measurement.
    """

    if not run_id:
        raise ValueError("run_id is required")
    prepared: dict[str, tuple[FeatureSpec, list[float], list[float]]] = {}
    for spec in specs:
        spec.validate()
        leakage_policy.validate(feature_name=spec.name, source=spec.source)
        if spec.name in prepared:
            raise ValueError(f"duplicate feature spec: {spec.name}")
        timestamps, values = _validated_samples(spec.name, samples_by_feature.get(spec.name, ()))
        prepared[spec.name] = (spec, timestamps, values)

    rows: list[dict[str, object]] = []
    previous_decision = float("-inf")
    for decision_index, raw_time in enumerate(decision_times):
        decision_time = float(raw_time)
        if not math.isfinite(decision_time) or decision_time <= previous_decision:
            raise ValueError("decision times must be finite and strictly increasing")
        row: dict[str, object] = {
            "run_id": run_id,
            "decision_index": decision_index,
            "decision_time": decision_time,
        }
        for name, (spec, timestamps, values) in prepared.items():
            selected = bisect_right(timestamps, decision_time) - 1
            if selected < 0:
                value, age, missing, source_time = 0.0, spec.maximum_age_seconds, 1, None
            else:
                source_time = timestamps[selected]
                age = decision_time - source_time
                if age < 0:
                    raise AssertionError(f"future sample selected for {name}")
                if age > spec.maximum_age_seconds:
                    value, age, missing, source_time = 0.0, spec.maximum_age_seconds, 1, None
                else:
                    value, missing = values[selected], 0
            row[name] = value
            row[f"{name}__age_seconds"] = age
            row[f"{name}__missing"] = missing
            # Audit-only source timestamps are stripped by the CSV exporter after its
            # timestamp assertion; they are never part of a deployable model matrix.
            row[f"__audit_source_time__{name}"] = source_time
        rows.append(row)
        previous_decision = decision_time
    return rows


"""Conservative temporal detectors for label-only operational failure evidence.

These functions operate on already time-aligned, label-only evidence. They do not
consume injection metadata and they return the first time at which the complete frozen
event definition has been observed. A gap larger than ``maximum_gap_seconds`` breaks a
claim of continuous evidence instead of being silently bridged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class PoseErrorSample:
    timestamp: float
    translation_error_m: float
    yaw_error_rad: float


@dataclass(frozen=True)
class MotionSample:
    timestamp: float
    commanded_linear_speed_mps: float
    odom_x_m: float
    odom_y_m: float


@dataclass(frozen=True)
class PerceptionSafetySample:
    timestamp: float
    obstacle_distance_from_footprint_m: float
    critical_obstacle_missed: bool
    emergency_intervention: bool


def _validate_times(samples: Sequence[object]) -> None:
    previous = float("-inf")
    for sample in samples:
        timestamp = float(getattr(sample, "timestamp"))
        if not math.isfinite(timestamp):
            raise ValueError("sample timestamps must be finite")
        if timestamp <= previous:
            raise ValueError("sample timestamps must be strictly increasing")
        previous = timestamp


def first_localisation_loss(
    samples: Sequence[PoseErrorSample],
    *,
    translation_threshold_m: float = 0.50,
    yaw_threshold_rad: float = 0.50,
    persistence_seconds: float = 2.0,
    maximum_gap_seconds: float = 2.5,
) -> float | None:
    """Return the first confirmed detection time for persistent localisation loss.

    Translation and yaw have independent persistence clocks because the approved rule
    is that either channel must continuously exceed its own threshold.
    """

    _validate_times(samples)
    if min(translation_threshold_m, yaw_threshold_rad, persistence_seconds,
           maximum_gap_seconds) <= 0:
        raise ValueError("thresholds, persistence, and maximum gap must be positive")
    translation_since: float | None = None
    yaw_since: float | None = None
    previous_time: float | None = None
    for sample in samples:
        t = float(sample.timestamp)
        if previous_time is not None and t - previous_time > maximum_gap_seconds:
            translation_since = yaw_since = None
        translation_since = (
            translation_since if sample.translation_error_m > translation_threshold_m
            and translation_since is not None
            else t if sample.translation_error_m > translation_threshold_m
            else None
        )
        yaw_since = (
            yaw_since if abs(sample.yaw_error_rad) > yaw_threshold_rad
            and yaw_since is not None
            else t if abs(sample.yaw_error_rad) > yaw_threshold_rad
            else None
        )
        if translation_since is not None and t - translation_since >= persistence_seconds:
            return t
        if yaw_since is not None and t - yaw_since >= persistence_seconds:
            return t
        previous_time = t
    return None


def first_immobilisation(
    samples: Sequence[MotionSample],
    *,
    command_threshold_mps: float = 0.05,
    maximum_progress_m: float = 0.50,
    persistence_seconds: float = 10.0,
    maximum_gap_seconds: float = 0.5,
) -> float | None:
    """Return the first time commanded linear motion lacks sufficient progress."""

    _validate_times(samples)
    if min(command_threshold_mps, maximum_progress_m, persistence_seconds,
           maximum_gap_seconds) <= 0:
        raise ValueError("thresholds, persistence, and maximum gap must be positive")
    candidate_time: float | None = None
    candidate_position: tuple[float, float] | None = None
    previous_time: float | None = None
    for sample in samples:
        t = float(sample.timestamp)
        position = (float(sample.odom_x_m), float(sample.odom_y_m))
        if not all(math.isfinite(value) for value in position):
            raise ValueError("odometry positions must be finite")
        command_active = abs(sample.commanded_linear_speed_mps) >= command_threshold_mps
        gap = previous_time is not None and t - previous_time > maximum_gap_seconds
        if gap or not command_active:
            candidate_time = candidate_position = None
        if command_active and candidate_time is None:
            candidate_time, candidate_position = t, position
        assert (candidate_time is None) == (candidate_position is None)
        if candidate_position is not None:
            progress = math.dist(position, candidate_position)
            if progress >= maximum_progress_m:
                # This sample proves that the previous candidate interval progressed.
                # It may also begin a new continuously-commanded candidate interval.
                candidate_time, candidate_position = t, position
            elif t - candidate_time >= persistence_seconds:
                return t
        previous_time = t
    return None


def first_unsafe_perception(
    samples: Sequence[PerceptionSafetySample],
    *,
    protected_region_m: float = 0.42,
) -> float | None:
    """Return the first fully confirmed unsafe-perception event time."""

    _validate_times(samples)
    if protected_region_m <= 0:
        raise ValueError("protected region must be positive")
    for sample in samples:
        if (
            sample.obstacle_distance_from_footprint_m <= protected_region_m
            and sample.critical_obstacle_missed
            and sample.emergency_intervention
        ):
            return float(sample.timestamp)
    return None


def first_primary_event(
    event_times: dict[str, float | None], precedence: Sequence[str]
) -> tuple[str, float] | None:
    """Select the earliest event, resolving exact ties by frozen precedence."""

    rank = {name: index for index, name in enumerate(precedence)}
    available: list[tuple[str, float]] = []
    for event_class, raw_time in event_times.items():
        if raw_time is None:
            continue
        event_time = float(raw_time)
        if not math.isfinite(event_time):
            raise ValueError(f"{event_class}: event time must be finite")
        available.append((event_class, event_time))
    if not available:
        return None
    return min(available, key=lambda item: (item[1], rank.get(item[0], len(rank))))

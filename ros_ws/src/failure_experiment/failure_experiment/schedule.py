"""Deterministic, fail-closed onset scheduling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FaultSchedule:
    clean_prefix_seconds: float
    planned_onset_seconds: float
    maximum_duration_seconds: float
    maximum_wait_seconds: float
    armed_time: float | None = None
    actual_onset_time: float | None = None
    rejection_time: float | None = None

    def __post_init__(self) -> None:
        if self.clean_prefix_seconds < 0:
            raise ValueError("clean prefix cannot be negative")
        if self.planned_onset_seconds < self.clean_prefix_seconds:
            raise ValueError("planned onset must not precede the clean prefix")
        if self.maximum_duration_seconds <= 0 or self.maximum_wait_seconds < 0:
            raise ValueError("duration must be positive and wait must be nonnegative")

    def arm(self, now: float) -> bool:
        if self.armed_time is not None:
            return False
        self.armed_time = float(now)
        return True

    def elapsed(self, now: float) -> float | None:
        return None if self.armed_time is None else float(now) - self.armed_time

    def update(self, now: float, *, eligible: bool) -> str:
        if self.armed_time is None:
            return "waiting_for_goal"
        elapsed = self.elapsed(now)
        assert elapsed is not None
        if self.actual_onset_time is not None:
            if now <= self.actual_onset_time + self.maximum_duration_seconds:
                return "active"
            return "complete"
        if self.rejection_time is not None:
            return "ineligible"
        if elapsed < self.planned_onset_seconds:
            return "clean_prefix" if elapsed < self.clean_prefix_seconds else "waiting_for_onset"
        if eligible:
            self.actual_onset_time = float(now)
            return "activated"
        if elapsed > self.planned_onset_seconds + self.maximum_wait_seconds:
            self.rejection_time = float(now)
            return "rejected"
        return "waiting_for_eligibility"


"""Threshold, persistence, and cooldown logic for warning scores."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence


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


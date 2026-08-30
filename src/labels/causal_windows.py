"""Construct past-only decision-window labels from episode annotations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class LabelConfig:
    history_seconds: float = 5.0
    warning_horizon_seconds: float = 10.0
    too_late_guard_seconds: float = 1.0
    negative_guard_seconds: float = 20.0
    decision_stride_seconds: float = 0.5

    def validate(self) -> None:
        if self.history_seconds <= 0:
            raise ValueError("history_seconds must be positive")
        if self.warning_horizon_seconds <= self.too_late_guard_seconds:
            raise ValueError("warning horizon must exceed the too-late guard")
        if self.too_late_guard_seconds <= 0 or self.negative_guard_seconds <= 0:
            raise ValueError("guard durations must be positive")
        if self.decision_stride_seconds <= 0:
            raise ValueError("decision_stride_seconds must be positive")


def _decision_times(start: float, end: float, history: float, stride: float) -> list[float]:
    first = Decimal(str(start)) + Decimal(str(history))
    final = Decimal(str(end))
    step = Decimal(str(stride))
    values: list[float] = []
    current = first
    while current <= final:
        values.append(float(current))
        current += step
    return values


def _primary_terminal_event(
    events: Sequence[Mapping[str, object]], precedence: Sequence[str]
) -> Mapping[str, object] | None:
    terminal = [event for event in events if event.get("terminal") is True]
    if not terminal:
        return None
    rank = {name: index for index, name in enumerate(precedence)}
    return min(
        terminal,
        key=lambda event: (
            float(event["time"]),
            rank.get(str(event.get("class")), len(rank)),
        ),
    )


def label_decision_times(
    *,
    episode_start: float,
    episode_end: float,
    events: Sequence[Mapping[str, object]],
    injection_onsets: Iterable[float],
    precedence: Sequence[str],
    config: LabelConfig,
) -> list[dict[str, object]]:
    """Return one auditable label record per decision time.

    Positive assignment takes precedence over the negative guard because injected
    onset can legitimately occur inside a causal precursor interval. Ground truth,
    event details, and injection metadata belong only in this label table.
    """

    config.validate()
    if episode_end < episode_start:
        raise ValueError("episode_end must not precede episode_start")

    primary = _primary_terminal_event(events, precedence)
    event_time = float(primary["time"]) if primary else None
    event_class = str(primary["class"]) if primary else None
    all_guard_times = [float(event["time"]) for event in events]
    all_guard_times.extend(float(value) for value in injection_onsets)

    records: list[dict[str, object]] = []
    for index, decision_time in enumerate(
        _decision_times(
            episode_start,
            episode_end,
            config.history_seconds,
            config.decision_stride_seconds,
        )
    ):
        label: int | None
        eligibility: str

        if event_time is not None and decision_time > event_time:
            label, eligibility = None, "excluded_post_event"
        elif (
            event_time is not None
            and event_time - config.too_late_guard_seconds < decision_time <= event_time
        ):
            label, eligibility = None, "excluded_too_late"
        elif (
            event_time is not None
            and event_time - config.warning_horizon_seconds
            <= decision_time
            <= event_time - config.too_late_guard_seconds
        ):
            label, eligibility = 1, "eligible_positive"
        elif all(
            abs(decision_time - protected_time) >= config.negative_guard_seconds
            for protected_time in all_guard_times
        ):
            label, eligibility = 0, "eligible_negative"
        else:
            label, eligibility = None, "excluded_near_event_or_injection"

        records.append(
            {
                "decision_index": index,
                "decision_time": decision_time,
                "window_start": decision_time - config.history_seconds,
                "window_end": decision_time,
                "label": label,
                "eligibility": eligibility,
                "primary_event_class": event_class,
                "primary_event_time": event_time,
            }
        )

    return records


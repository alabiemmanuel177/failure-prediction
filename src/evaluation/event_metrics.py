"""Event-level early-warning metrics; windows are never treated as independent."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def _truth(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def evaluate_event_warnings(episodes: Mapping[str, Sequence[Mapping[str, object]]]) -> dict:
    event_count = detected_count = false_alerts = non_event_missions = 0
    false_alerts_non_event = 0
    lead_times: list[float] = []
    per_episode = []
    for run_id, rows in sorted(episodes.items()):
        if not rows:
            raise ValueError(f"{run_id}: empty episode rows")
        event_times = {
            float(row["primary_event_time"])
            for row in rows if row.get("primary_event_time") not in {None, "", "None"}
        }
        if len(event_times) > 1:
            raise ValueError(f"{run_id}: inconsistent primary_event_time")
        event_time = next(iter(event_times)) if event_times else None
        useful = sorted(
            float(row["decision_time"]) for row in rows
            if _truth(row.get("alarm")) and row.get("eligibility") == "eligible_positive"
        )
        episode_false = sum(
            1 for row in rows
            if _truth(row.get("alarm")) and row.get("eligibility") == "eligible_negative"
        )
        false_alerts += episode_false
        detected = bool(useful)
        lead_time = None
        if event_time is None:
            non_event_missions += 1
            false_alerts_non_event += episode_false
        else:
            event_count += 1
            if detected:
                detected_count += 1
                lead_time = event_time - useful[0]
                if lead_time < 0:
                    raise AssertionError("negative warning lead time")
                lead_times.append(lead_time)
        per_episode.append({
            "run_id": run_id, "has_event": event_time is not None,
            "detected": detected, "first_useful_lead_seconds": lead_time,
            "false_alerts": episode_false,
        })
    mission_count = len(episodes)
    sorted_leads = sorted(lead_times)
    median = None
    if sorted_leads:
        middle = len(sorted_leads) // 2
        median = sorted_leads[middle] if len(sorted_leads) % 2 else (
            sorted_leads[middle - 1] + sorted_leads[middle]
        ) / 2
    return {
        "episode_count": mission_count,
        "event_count": event_count,
        "detected_event_count": detected_count,
        "undetected_event_count": event_count - detected_count,
        "event_recall": detected_count / event_count if event_count else None,
        "false_alert_count": false_alerts,
        "false_alerts_per_mission": false_alerts / mission_count if mission_count else math.nan,
        "non_event_mission_count": non_event_missions,
        "false_alerts_per_non_event_mission": (
            false_alerts_non_event / non_event_missions if non_event_missions else None
        ),
        "median_useful_lead_seconds_detected": median,
        "lead_time_denominator_detected": len(lead_times),
        "per_episode": per_episode,
    }

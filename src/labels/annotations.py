"""Validation for human-authored episode event annotations."""

from __future__ import annotations

from typing import Any, Mapping


def validate_annotation(annotation: Mapping[str, Any], taxonomy: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    episode = annotation.get("episode", {})
    try:
        start = float(episode["start_time"])
        end = float(episode["end_time"])
        if end < start:
            errors.append("episode.end_time precedes start_time")
    except (KeyError, TypeError, ValueError):
        errors.append("episode start_time and end_time must be numeric")
        start = end = 0.0

    known_families = taxonomy.get("fault_families", {})
    for index, injection in enumerate(annotation.get("injections", [])):
        family = injection.get("family")
        if family not in known_families:
            errors.append(f"injections[{index}].family is unknown: {family!r}")
            continue
        allowed = known_families[family].get("allowed_severities")
        severity = injection.get("severity")
        if allowed and severity not in allowed:
            errors.append(f"injections[{index}].severity must be one of {allowed}")
        onset = injection.get("actual_onset")
        if injection.get("eligible") is True and onset is None:
            errors.append(f"injections[{index}] is eligible but actual_onset is missing")
        if onset is not None and not start <= float(onset) <= end:
            errors.append(f"injections[{index}].actual_onset lies outside the episode")

    known_outcomes = taxonomy.get("terminal_outcomes", {})
    event_ids: set[str] = set()
    previous_time = float("-inf")
    for index, event in enumerate(annotation.get("events", [])):
        event_id = event.get("event_id")
        if not event_id:
            errors.append(f"events[{index}].event_id is required")
        elif event_id in event_ids:
            errors.append(f"events[{index}].event_id is duplicated: {event_id}")
        else:
            event_ids.add(event_id)
        event_class = event.get("class")
        if event_class not in known_outcomes:
            errors.append(f"events[{index}].class is unknown: {event_class!r}")
        if event.get("terminal") is not True:
            errors.append(f"events[{index}] uses a terminal outcome class but terminal is not true")
        try:
            event_time = float(event["time"])
            if not start <= event_time <= end:
                errors.append(f"events[{index}].time lies outside the episode")
            if event_time < previous_time:
                errors.append("events must be ordered by nondecreasing time")
            previous_time = event_time
        except (KeyError, TypeError, ValueError):
            errors.append(f"events[{index}].time must be numeric")
        if event.get("confidence") not in {"confirmed", "ambiguous"}:
            errors.append(f"events[{index}].confidence must be confirmed or ambiguous")
        if not event.get("confirmation_sources"):
            errors.append(f"events[{index}].confirmation_sources must not be empty")

    exclusion = annotation.get("episode_exclusion", {})
    if exclusion.get("excluded") is True:
        allowed_reasons = {
            name
            for name, rule in taxonomy.get("exclusion_reasons", {}).items()
            if rule.get("exclude_episode") is True
        }
        if exclusion.get("reason") not in allowed_reasons:
            errors.append("episode_exclusion.reason is not a permitted episode exclusion")

    return errors


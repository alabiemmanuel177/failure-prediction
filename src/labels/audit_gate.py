"""Exact automatic-versus-human-review agreement checks for Gate G1."""

from __future__ import annotations

from typing import Any, Mapping

from .annotations import validate_annotation


def causal_signature(annotation: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields that can change event or window labels."""
    return {
        "episode": {
            key: annotation.get("episode", {}).get(key)
            for key in ("start_time", "end_time", "termination_reason")
        },
        "injections": [
            {
                key: injection.get(key)
                for key in (
                    "family", "severity", "planned_onset", "actual_onset",
                    "duration_seconds", "eligible",
                )
            }
            for injection in annotation.get("injections", [])
        ],
        "events": [
            {
                key: event.get(key)
                for key in ("event_id", "class", "time", "terminal")
            }
            for event in annotation.get("events", [])
        ],
        "episode_exclusion": {
            key: annotation.get("episode_exclusion", {}).get(key)
            for key in ("excluded", "reason")
        },
    }


def validate_review_pair(
    automatic: Mapping[str, Any],
    reviewed: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
) -> list[str]:
    errors = validate_annotation(reviewed, taxonomy)
    if automatic.get("run_id") != reviewed.get("run_id"):
        errors.append("reviewed run_id differs from automatic annotation")
    review = reviewed.get("review", {})
    first = review.get("first_reviewer")
    second = review.get("second_reviewer")
    if not first:
        errors.append("review.first_reviewer is required")
    if not second:
        errors.append("review.second_reviewer is required")
    if first and second and first == second:
        errors.append("first and second reviewers must be different people")
    if review.get("adjudication_status") not in {"agreed", "adjudicated"}:
        errors.append("review.adjudication_status must be agreed or adjudicated")
    if causal_signature(automatic) != causal_signature(reviewed):
        errors.append("human-reviewed causal fields do not exactly match automatic extraction")
    return errors


def validate_primary_review(
    automatic: Mapping[str, Any],
    reviewed: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
) -> list[str]:
    """Validate the Protocol 1.1 single-reviewer admission requirement.

    A second reviewer remains useful optional evidence, but is not required for model
    development.  This validator deliberately makes no inter-rater-agreement claim.
    """
    errors = validate_annotation(reviewed, taxonomy)
    if automatic.get("run_id") != reviewed.get("run_id"):
        errors.append("reviewed run_id differs from automatic annotation")
    review = reviewed.get("review", {})
    if not review.get("first_reviewer"):
        errors.append("review.first_reviewer is required")
    if not review.get("first_reviewed_utc"):
        errors.append("review.first_reviewed_utc is required")
    if causal_signature(automatic) != causal_signature(reviewed):
        errors.append("human-reviewed causal fields do not exactly match automatic extraction")
    return errors

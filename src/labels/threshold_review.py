"""Validation for prospective terminal-event threshold approvals."""

from __future__ import annotations

from typing import Any, Mapping


EXPECTED_FIELDS = {
    "localisation_loss": (
        "translation_error_m", "yaw_error_rad", "persistence_seconds",
        "threshold_logic",
    ),
    "immobilisation": (
        "minimum_command_speed_mps", "maximum_progress_m", "persistence_seconds",
        "angular_only_commands_count", "progress_measure",
    ),
    "unsafe_perception": (
        "protected_stopping_region_m_from_footprint",
        "added_braking_or_latency_margin_m",
    ),
}


def validate_researcher_threshold_review(review: Mapping[str, Any]) -> list[str]:
    """Validate the prospectively recorded researcher approval used by Protocol 1.1."""
    findings: list[str] = []
    if str(review.get("reviewer_role", "")).casefold() != "researcher":
        findings.append("reviewer_role must be Researcher")
    reviewer = str(review.get("reviewer_name", ""))
    if not reviewer or reviewer.startswith("TODO"):
        findings.append("reviewer_name is unresolved")
    reviewed_utc = str(review.get("reviewed_utc", ""))
    if not reviewed_utc or reviewed_utc.startswith("TODO"):
        findings.append("reviewed_utc is unresolved")
    if review.get("protected_outcomes_consulted") is not False:
        findings.append("protected_outcomes_consulted must be false")
    decisions = review.get("decisions", {})
    if set(decisions) != set(EXPECTED_FIELDS):
        findings.append("review must contain exactly the three prespecified event decisions")
    for event, fields in EXPECTED_FIELDS.items():
        values = decisions.get(event, {})
        if values.get("decision") != "accept":
            findings.append(f"{event}: researcher decision must be accept")
        rationale = str(values.get("rationale", ""))
        if not rationale or rationale.startswith("TODO"):
            findings.append(f"{event}: rationale is unresolved")
        for field in fields:
            if field not in values:
                findings.append(f"{event}.{field} is missing")
    if review.get("overall_decision") != "approved":
        findings.append("overall_decision must be approved")
    if review.get("requires_existing_development_data_invalidation") is not False:
        findings.append("accepted values must not invalidate existing development data")
    return findings


def validate_threshold_review(
    review: Mapping[str, Any],
    approved_values: Mapping[str, Any],
    *,
    researcher_name: str | None = None,
) -> list[str]:
    findings: list[str] = []
    if str(review.get("reviewer_role", "")).casefold() != "supervisor":
        findings.append("reviewer_role must be Supervisor")
    reviewer = str(review.get("reviewer_name", ""))
    if not reviewer or reviewer.startswith("TODO"):
        findings.append("reviewer_name is unresolved")
    if researcher_name and reviewer.casefold() == researcher_name.casefold():
        findings.append("supervisor reviewer must be distinct from the researcher")
    reviewed_utc = str(review.get("reviewed_utc", ""))
    if not reviewed_utc or reviewed_utc.startswith("TODO"):
        findings.append("reviewed_utc is unresolved")
    if review.get("protected_outcomes_consulted") is not False:
        findings.append("protected_outcomes_consulted must be false")
    decisions = review.get("decisions", {})
    if set(decisions) != set(EXPECTED_FIELDS):
        findings.append("review must contain exactly the three prespecified event decisions")
    for event, fields in EXPECTED_FIELDS.items():
        values = decisions.get(event, {})
        if values.get("decision") != "accept":
            findings.append(f"{event}: supervisor decision must be accept")
        rationale = str(values.get("rationale", ""))
        if not rationale or rationale.startswith("TODO"):
            findings.append(f"{event}: rationale is unresolved")
        expected = approved_values.get(event, {})
        for field in fields:
            if values.get(field) != expected.get(field):
                findings.append(
                    f"{event}.{field} differs from the prospectively approved values"
                )
    if review.get("overall_decision") != "approved":
        findings.append("overall_decision must be approved")
    if review.get("requires_protocol_version_change") is not False:
        findings.append("accepted values must not require a protocol version change")
    if review.get("requires_existing_development_data_invalidation") is not False:
        findings.append("accepted values must not invalidate existing development data")
    return findings

"""Conservative, evidence-backed completion audit for the full Research 2 objective."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _campaign_complete(root: Path, relative: str, expected: int) -> bool:
    report = _yaml(root / relative) or {}
    counts = report.get("counts", {})
    integrity = report.get("integrity", {})
    return bool(
        report.get("complete_and_artifact_valid") is True
        and report.get("protected_test_used") is False
        and counts.get("expected") == expected
        and counts.get("observed") == expected
        and counts.get("usable") == expected
        and all(
            integrity.get(key) == []
            for key in (
                "missing_episode_keys", "unexpected_episode_keys",
                "duplicate_episode_keys", "replacement_errors",
            )
        )
    )


def _dataset_complete(
    root: Path, relative: str, inventory_relative: str,
    dataset_id: str, expected: int,
) -> bool:
    document = _yaml(root / relative) or {}
    inventory = root / inventory_relative
    return bool(
        document.get("dataset_id") == dataset_id
        and document.get("protected_test_used") is False
        and document.get("episode_inventory", {}).get("rows") == expected
        and isinstance(document.get("episode_inventory", {}).get("sha256"), str)
        and len(document["episode_inventory"]["sha256"]) == 64
        and inventory.is_file()
        and sum(1 for line in inventory.read_text(encoding="utf-8").splitlines()
                if line.strip()) == expected
    )


def audit_completion(root: Path) -> list[str]:
    findings: list[str] = []
    amendment = _yaml(root / "configs/protocol_amendment_1.1.yaml")
    researcher_review = _yaml(
        root / "configs/event_threshold_review.2026-08-30.researcher.yaml"
    )
    if not amendment or amendment.get("status") != "approved":
        findings.append("Protocol 1.1 human-review amendment is missing")
    if not researcher_review or researcher_review.get("overall_decision") != "approved":
        findings.append("researcher threshold approval is missing")
    reviewed = list((root / "data/annotations/reviewed").glob("*.yaml"))
    complete_reviews = 0
    for path in reviewed:
        document = _yaml(path)
        review = document.get("review", {}) if document else {}
        first = str(review.get("first_reviewer") or "").strip()
        if first and review.get("first_reviewed_utc"):
            complete_reviews += 1
    if complete_reviews != 20:
        findings.append(f"primary human-reviewed annotations: {complete_reviews}/20")

    pilot_relative = "reports/pilot/balanced_pilot_v1.cumulative648.yaml"
    if not _campaign_complete(root, pilot_relative, 648):
        findings.append(f"balanced pilot evidence missing or invalid: {pilot_relative}")

    campaign_requirements = [
        ("validation campaign", "reports/validation/balanced_validation_v1.cumulative324.yaml", 324),
        ("targeted development campaign", "reports/pilot/targeted_development_v1.cumulative1212.yaml", 1212),
    ]
    for label, relative, expected in campaign_requirements:
        if not _campaign_complete(root, relative, expected):
            findings.append(f"{label} evidence missing or invalid: {relative}")

    dataset_requirements = [
        (
            "development dataset", "data/manifests/balanced_pilot_v1.dataset.yaml",
            "data/manifests/balanced_pilot_v1.episodes.jsonl",
            "balanced_pilot_v1-development-648", 648,
        ),
        (
            "validation dataset", "data/manifests/balanced_validation_v1.dataset.yaml",
            "data/manifests/balanced_validation_v1.episodes.jsonl",
            "balanced_validation_v1-validation-324", 324,
        ),
        (
            "targeted development dataset", "data/manifests/targeted_development_v1.dataset.yaml",
            "data/manifests/targeted_development_v1.episodes.jsonl",
            "targeted_development_v1-development-1212", 1212,
        ),
    ]
    for label, relative, inventory, dataset_id, expected in dataset_requirements:
        if not _dataset_complete(root, relative, inventory, dataset_id, expected):
            findings.append(f"{label} inventory missing or invalid: {relative}")

    raw_payload = _yaml(root / "reports/integrity/raw_payload_integrity.yaml") or {}
    if not (
        raw_payload.get("passed") is True
        and raw_payload.get("protected_outcomes_consulted") is False
        and raw_payload.get("counts", {}).get("unexpected_zero_length_runs") == 0
    ):
        findings.append("raw artifact payload integrity evidence missing or invalid")

    feature_smoke = _yaml(root / "reports/integrity/offline_feature_pipeline_smoke_v1.yaml") or {}
    feature_checks = feature_smoke.get("checks", {})
    if not (
        feature_smoke.get("status") == "passed"
        and feature_smoke.get("protected_test_used") is False
        and feature_smoke.get("training_performed") is False
        and feature_checks.get("all_primary_value_age_missing_columns_present") is True
        and feature_checks.get("future_samples_rejected_by_extractor") is True
    ):
        findings.append("offline causal feature-pipeline evidence missing or invalid")

    sequence_smoke = _yaml(root / "reports/integrity/causal_sequence_pipeline_smoke_v1.yaml") or {}
    sequence_checks = sequence_smoke.get("checks", {})
    if not (
        sequence_smoke.get("status") == "passed"
        and sequence_smoke.get("protected_test_used") is False
        and sequence_smoke.get("training_performed") is False
        and sequence_checks.get("every_sequence_has_complete_history") is True
        and sequence_checks.get("every_raw_source_timestamp_not_after_sample_time") is True
        and sequence_checks.get("future_sample_mutation_invariance_test") == "passed"
    ):
        findings.append("causal temporal-sequence evidence missing or invalid")

    positive_smoke = _yaml(root / "reports/integrity/positive_sequence_pipeline_smoke_v1.yaml") or {}
    positive_checks = positive_smoke.get("checks", {})
    if not (
        positive_smoke.get("status") == "passed"
        and positive_smoke.get("protected_test_used") is False
        and positive_smoke.get("training_performed") is False
        and positive_checks.get("no_too_late_window_emitted") is True
        and positive_checks.get("last_lead_exceeds_one_second_guard") is True
    ):
        findings.append("positive warning-window sequence evidence missing or invalid")

    guards = _yaml(root / "reports/recovery/guard_verification.yaml") or {}
    if not (
        guards.get("passed") is True
        and guards.get("protected_test_used") is False
        and guards.get("violation_count") == 0
    ):
        findings.append("recovery guard verification evidence missing or invalid")

    required_yaml = {
        "model freeze": ("configs/model_freeze.yaml", "frozen"),
        "held-out map result": ("reports/confirmatory/held_out_map.yaml", "complete"),
        "unseen-family result": ("reports/confirmatory/unseen_family.yaml", "complete"),
        "paired recovery result": ("reports/recovery/paired_recovery.yaml", "complete"),
        "independent reproduction": ("reports/reproduction/release.yaml", "passed"),
    }
    for label, (relative, flag) in required_yaml.items():
        document = _yaml(root / relative)
        if not document or document.get(flag) is not True:
            findings.append(f"{label} evidence missing or not {flag}: {relative}")

    model_card = root / "docs/model-card.md"
    if not model_card.exists() or "Status: final" not in model_card.read_text(encoding="utf-8"):
        findings.append("final model card is missing")
    dataset_card = root / "docs/dataset-card.md"
    if not dataset_card.exists() or "Status: final" not in dataset_card.read_text(encoding="utf-8"):
        findings.append("final dataset card is missing")
    manuscript = root / "manuscript/main.md"
    manuscript_text = manuscript.read_text(encoding="utf-8") if manuscript.exists() else ""
    if not manuscript_text:
        findings.append("manuscript is missing")
    elif "RESULT_PENDING" in manuscript_text or "RELEASE_PENDING" in manuscript_text:
        findings.append("manuscript still contains pending result or release markers")
    return findings

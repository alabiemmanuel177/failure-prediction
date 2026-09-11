"""Evidence-backed audit of work that must precede model development.

The audit deliberately separates machine-executable engineering from approvals that
must be supplied by distinct people.  It never treats a missing model, protected-map
result, or paired post-model recovery trial as a non-model defect.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _review_counts(root: Path) -> tuple[int, int]:
    first = second = 0
    for path in (root / "data/annotations/reviewed").glob("*.yaml"):
        document = _yaml(path) or {}
        review = document.get("review", {})
        if str(review.get("first_reviewer") or "").strip():
            first += 1
        if str(review.get("second_reviewer") or "").strip():
            second += 1
    return first, second


def _research_log_valid(path: Path) -> bool:
    if not path.exists():
        return False
    previous = "GENESIS"
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            stored = record.pop("record_hash")
            if record.get("previous_hash") != previous:
                return False
            payload = json.dumps(
                record, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            computed = hashlib.sha256(payload).hexdigest()
            if stored != computed:
                return False
            previous = stored
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _research1_catalog_valid(root: Path) -> bool:
    report = _yaml(root / "reports/integrity/research1_temporal_catalog_v2.yaml") or {}
    path = root / "data/manifests/research1_temporal_catalog_v2.jsonl"
    if not path.is_file():
        return False
    try:
        payload = path.read_bytes()
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    forbidden = {"terminal_state", "success", "collision", "timeout", "shift_family", "severity"}
    return bool(
        report.get("schema_version") == 2
        and report.get("counts", {}).get("catalog_rows") == len(rows) == 1189
        and report.get("counts", {}).get("by_split")
        == {"development": 825, "validation": 364}
        and report.get("catalog", {}).get("sha256")
        == hashlib.sha256(payload).hexdigest()
        and report.get("controls", {}).get("protected_test_rows_exported") == 0
        and report.get("controls", {}).get("nonrequired_topic_names_exported") is False
        and report.get("controls", {}).get("admitted_for_model_use") is False
        and all(row.get("split") in {"development", "validation"} for row in rows)
        and all(not (forbidden & set(row)) for row in rows)
    )


def audit_nonmodel(root: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {}

    faults = list((root / "configs/faults").glob("*.yaml"))
    checks["seven_fault_families_frozen"] = len(faults) == 7 and all(
        (_yaml(path) or {}).get("status") == "frozen" for path in faults
    )
    fault_gate = _yaml(root / "reports/integrity/fault_integrity_gate_v1.yaml") or {}
    checks["fault_treatment_integrity_21_of_21"] = (
        fault_gate.get("status") == "passed"
        or fault_gate.get("passed") is True
    ) and fault_gate.get("passed_cells") == fault_gate.get("matrix_size") == 21

    pilot = _yaml(root / "reports/pilot/balanced_pilot_v1.cumulative648.yaml") or {}
    pilot_counts = pilot.get("counts", {})
    checks["development_pilot_648_complete"] = bool(
        pilot.get("complete_and_artifact_valid") is True
        and pilot.get("protected_test_used") is False
        and pilot_counts.get("expected") == pilot_counts.get("usable") == 648
    )
    dataset = _yaml(root / "data/manifests/balanced_pilot_v1.dataset.yaml") or {}
    inventory_path = root / "data/manifests/balanced_pilot_v1.episodes.jsonl"
    checks["development_inventory_hash_addressed"] = bool(
        dataset.get("dataset_id") == "balanced_pilot_v1-development-648"
        and dataset.get("protected_test_used") is False
        and dataset.get("episode_inventory", {}).get("rows") == 648
        and inventory_path.exists()
    )
    sizing = _yaml(root / "reports/pilot/post_pilot_campaign_size_v2.yaml") or {}
    sizing_correction = _yaml(
        root / "reports/pilot/training_volume_split_correction_v1.yaml"
    ) or {}
    targeted_manifest = _yaml(root / "data/manifests/targeted_development_v1.yaml") or {}
    checks["post_pilot_campaign_size_frozen"] = bool(
        sizing.get("protected_outcomes_consulted") is False
        and sizing.get("schema_version") == 2
        and sizing.get("recommended_targeted_addition_episodes") == 1212
        and sizing.get(
            "maximum_fitting_episodes_before_supplement_if_all_research1_development_admitted"
        ) == 2685
        and sizing.get("minimum_route_balanced_development_supplement") == 324
        and sizing.get("inputs", {}).get(
            "research1_development_structurally_eligible_bags"
        ) == 825
        and sizing.get("inputs", {}).get(
            "research1_validation_reserved_for_selection"
        ) == 364
        and sizing.get("interpretation", {}).get(
            "research1_validation_is_selection_only"
        ) is True
        and sizing_correction.get("protected_outcomes_consulted") is False
        and sizing_correction.get("corrected_accounting", {}).get(
            "maximum_fitting_episodes_before_supplement"
        ) == 2685
        and sizing_correction.get("corrected_accounting", {}).get(
            "research1_validation_reserved_for_selection"
        ) == 364
        and sizing_correction.get("supplement_rule", {}).get(
            "validation_or_test_episodes_may_count_toward_fitting"
        ) is False
    )
    checks["targeted_development_campaign_preregistered"] = bool(
        targeted_manifest.get("expected_episode_count") == 1212
        and targeted_manifest.get("allowed_splits") == ["development"]
        and targeted_manifest.get("protected_test_used") is False
        and targeted_manifest.get("planning_source")
        == "reports/pilot/post_pilot_campaign_size_v2.yaml"
    )
    targeted = _yaml(
        root / "reports/pilot/targeted_development_v1.cumulative1212.yaml"
    ) or {}
    targeted_counts = targeted.get("counts", {})
    checks["targeted_development_1212_complete"] = bool(
        targeted.get("complete_and_artifact_valid") is True
        and targeted.get("protected_test_used") is False
        and targeted_counts.get("expected") == targeted_counts.get("usable") == 1212
    )
    targeted_dataset = _yaml(
        root / "data/manifests/targeted_development_v1.dataset.yaml"
    ) or {}
    checks["targeted_development_inventory_hash_addressed"] = bool(
        targeted_dataset.get("dataset_id") == "targeted_development_v1-development-1212"
        and targeted_dataset.get("protected_test_used") is False
        and targeted_dataset.get("episode_inventory", {}).get("rows") == 1212
    )

    validation = _yaml(
        root / "reports/validation/balanced_validation_v1.cumulative324.yaml"
    ) or {}
    validation_counts = validation.get("counts", {})
    checks["validation_collection_324_complete"] = bool(
        validation.get("complete_and_artifact_valid") is True
        and validation.get("protected_test_used") is False
        and validation_counts.get("expected") == validation_counts.get("usable") == 324
    )
    validation_dataset = _yaml(
        root / "data/manifests/balanced_validation_v1.dataset.yaml"
    ) or {}
    checks["validation_inventory_hash_addressed"] = bool(
        validation_dataset.get("dataset_id") == "balanced_validation_v1-validation-324"
        and validation_dataset.get("protected_test_used") is False
        and validation_dataset.get("episode_inventory", {}).get("rows") == 324
    )

    recovery = _yaml(root / "reports/recovery/guard_verification.yaml") or {}
    checks["recovery_guards_exhaustively_verified"] = bool(
        recovery.get("passed") is True
        and recovery.get("protected_test_used") is False
        and recovery.get("live_execution_performed") is False
        and recovery.get("violation_count") == 0
    )
    reuse = _yaml(root / "reports/integrity/research1_reuse_compatibility_v2.yaml") or {}
    checks["research1_reuse_boundary_audited"] = bool(
        reuse.get("protected_outcomes_consulted") is False
        and reuse.get("counts", {}).get(
            "development_or_validation_bags_with_core_temporal_topics"
        ) == 1189
        and reuse.get("counts", {}).get("core_temporal_topics_by_split", {}).get(
            "development"
        ) is not None
        and reuse.get("counts", {}).get("core_temporal_topics_by_split", {}).get(
            "validation"
        ) is not None
    )
    checks["research1_structural_catalog_minimized"] = _research1_catalog_valid(root)
    checks["literature_protocol_review_complete"] = all(
        (root / relative).exists()
        for relative in (
            "literature/evidence-matrix.md",
            "literature/search-log.md",
            "literature/study-appraisal.md",
            "literature/references.bib",
        )
    )
    checks["research_log_hash_chain_valid"] = _research_log_valid(
        root / "logs/research-log.jsonl"
    )
    raw_payloads = _yaml(root / "reports/integrity/raw_payload_integrity.yaml") or {}
    checks["raw_payload_integrity_passes"] = bool(
        raw_payloads.get("passed") is True
        and raw_payloads.get("protected_outcomes_consulted") is False
        and raw_payloads.get("counts", {}).get("unexpected_zero_length_runs") == 0
    )

    split_manifest = _yaml(root / "data/manifests/splits.template.yaml") or {}
    protected = split_manifest.get("held_out_map_test", {})
    boundary_checks = {
        "protected_split_unassigned_and_unused": bool(
            not protected.get("maps") and not protected.get("routes")
        ),
        "no_model_or_confirmatory_artifact_created": not any(
            (root / relative).exists()
            for relative in (
                "configs/model_freeze.yaml",
                "reports/confirmatory/held_out_map.yaml",
                "reports/confirmatory/unseen_family.yaml",
            )
        ),
    }
    freeze_path = root / "configs/model_freeze.yaml"
    freeze = _yaml(freeze_path) or {}
    frozen = freeze.get("frozen") is True
    model_freeze = {
        "frozen": frozen,
        "path": "configs/model_freeze.yaml" if freeze_path.exists() else None,
        "sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest() if freeze_path.exists() else None,
        "frozen_utc": freeze.get("frozen_utc") if frozen else None,
    }
    post_freeze: dict[str, Any] = {}
    if frozen:
        # These boundaries hold only until the model freeze: the freeze record itself,
        # the post-freeze protected split assignment and confirmatory outputs are
        # expected afterwards, so the checks are reported as ``post_freeze`` with their
        # observed value instead of counting as incomplete pre-model work.
        post_freeze = {
            **boundary_checks,
            "status": "post_freeze",
            "expected_value_after_freeze": False,
            "note": (
                "pre-model boundary checks are informational after configs/model_freeze.yaml "
                "declares frozen: true; protected split assignment and confirmatory "
                "artifacts are expected post-freeze work"
            ),
        }
    else:
        checks.update(boundary_checks)

    first_reviews, second_reviews = _review_counts(root)
    human_requirements = {
        "first_reviewer_annotations": f"{first_reviews}/20",
        "second_distinct_reviewer_annotations": f"{second_reviews}/20",
        "researcher_threshold_approval": bool(
            (_yaml(root / "configs/event_threshold_review.2026-08-30.researcher.yaml") or {})
            .get("overall_decision") == "approved"
        ),
        "protocol_1_1_amendment_approved": bool(
            (_yaml(root / "configs/protocol_amendment_1.1.yaml") or {})
            .get("status") == "approved"
        ),
        "admission_rule": (
            "model fitting requires 20/20 primary human reviews with exact causal "
            "agreement and the prospectively recorded researcher threshold approval; "
            "independent review is optional and no inter-rater agreement is claimed"
        ),
    }
    incomplete = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "audit_scope": "pre_model_machine_executable_work",
        "status": "complete" if not incomplete else "incomplete",
        "protected_outcomes_consulted": False,
        "model_freeze": model_freeze,
        "checks": checks,
        "post_freeze": post_freeze,
        "incomplete_checks": incomplete,
        "external_human_requirements": human_requirements,
        "excluded_post_model_work": [
            "model fitting and calibration",
            "protected-map and unseen-family confirmatory evaluation",
            "prediction-triggered paired recovery campaign",
            "final model card, results, and release reproduction",
        ],
    }

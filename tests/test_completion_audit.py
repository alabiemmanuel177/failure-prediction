from pathlib import Path

import yaml

from src.completion import audit_completion


def write_yaml(root: Path, relative: str, value: dict):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def test_completion_audit_is_fail_closed(tmp_path):
    findings = audit_completion(tmp_path)
    assert "Protocol 1.1 human-review amendment is missing" in findings
    assert "researcher threshold approval is missing" in findings
    assert "primary human-reviewed annotations: 0/20" in findings
    assert "manuscript is missing" in findings


def test_completion_audit_accepts_only_full_evidence_contract(tmp_path):
    write_yaml(tmp_path, "configs/protocol_amendment_1.1.yaml", {"status": "approved"})
    write_yaml(tmp_path, "configs/event_threshold_review.2026-08-30.researcher.yaml", {
        "overall_decision": "approved",
    })
    reviewed = tmp_path / "data/annotations/reviewed"
    reviewed.mkdir(parents=True)
    for index in range(20):
        (reviewed / f"{index}.yaml").write_text(
            "review:\n"
            "  first_reviewer: Reviewer A\n"
            "  first_reviewed_utc: '2026-08-31T22:11:37Z'\n"
            "  second_reviewer: null\n"
            "  adjudication_status: pending\n",
            encoding="utf-8",
        )
    write_yaml(tmp_path, "reports/pilot/balanced_pilot_v1.cumulative648.yaml", {
        "complete_and_artifact_valid": True,
        "protected_test_used": False,
        "counts": {"expected": 648, "observed": 648, "usable": 648},
        "integrity": {
            "missing_episode_keys": [],
            "unexpected_episode_keys": [],
            "duplicate_episode_keys": [],
            "replacement_errors": [],
        },
    })
    for relative, expected in [
        ("reports/validation/balanced_validation_v1.cumulative324.yaml", 324),
        ("reports/pilot/targeted_development_v1.cumulative1212.yaml", 1212),
    ]:
        write_yaml(tmp_path, relative, {
            "complete_and_artifact_valid": True,
            "protected_test_used": False,
            "counts": {"expected": expected, "observed": expected, "usable": expected},
            "integrity": {
                "missing_episode_keys": [], "unexpected_episode_keys": [],
                "duplicate_episode_keys": [], "replacement_errors": [],
            },
        })
    for relative, inventory, dataset_id, expected in [
        (
            "data/manifests/balanced_pilot_v1.dataset.yaml",
            "data/manifests/balanced_pilot_v1.episodes.jsonl",
            "balanced_pilot_v1-development-648", 648,
        ),
        (
            "data/manifests/balanced_validation_v1.dataset.yaml",
            "data/manifests/balanced_validation_v1.episodes.jsonl",
            "balanced_validation_v1-validation-324", 324,
        ),
        (
            "data/manifests/targeted_development_v1.dataset.yaml",
            "data/manifests/targeted_development_v1.episodes.jsonl",
            "targeted_development_v1-development-1212", 1212,
        ),
    ]:
        inventory_path = tmp_path / inventory
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text("{}\n" * expected, encoding="utf-8")
        write_yaml(tmp_path, relative, {
            "dataset_id": dataset_id, "protected_test_used": False,
            "episode_inventory": {"rows": expected, "sha256": "a" * 64},
        })
    write_yaml(tmp_path, "reports/integrity/raw_payload_integrity.yaml", {
        "passed": True, "protected_outcomes_consulted": False,
        "counts": {"unexpected_zero_length_runs": 0},
    })
    write_yaml(tmp_path, "reports/integrity/offline_feature_pipeline_smoke_v1.yaml", {
        "status": "passed", "protected_test_used": False, "training_performed": False,
        "checks": {
            "all_primary_value_age_missing_columns_present": True,
            "future_samples_rejected_by_extractor": True,
        },
    })
    write_yaml(tmp_path, "reports/integrity/causal_sequence_pipeline_smoke_v1.yaml", {
        "status": "passed", "protected_test_used": False, "training_performed": False,
        "checks": {
            "every_sequence_has_complete_history": True,
            "every_raw_source_timestamp_not_after_sample_time": True,
            "future_sample_mutation_invariance_test": "passed",
        },
    })
    write_yaml(tmp_path, "reports/integrity/positive_sequence_pipeline_smoke_v1.yaml", {
        "status": "passed", "protected_test_used": False, "training_performed": False,
        "checks": {
            "no_too_late_window_emitted": True,
            "last_lead_exceeds_one_second_guard": True,
        },
    })
    write_yaml(tmp_path, "reports/recovery/guard_verification.yaml", {
        "passed": True, "protected_test_used": False, "violation_count": 0,
    })
    for relative, flag in [
        ("configs/model_freeze.yaml", "frozen"),
        ("reports/confirmatory/held_out_map.yaml", "complete"),
        ("reports/confirmatory/unseen_family.yaml", "complete"),
        ("reports/recovery/paired_recovery.yaml", "complete"),
        ("reports/reproduction/release.yaml", "passed"),
    ]:
        write_yaml(tmp_path, relative, {flag: True})
    for relative, text in [
        ("docs/model-card.md", "Status: final\n"),
        ("docs/dataset-card.md", "Status: final\n"),
        ("manuscript/main.md", "Finished manuscript.\n"),
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert audit_completion(tmp_path) == []


def test_completion_audit_rejects_incomplete_pilot_contract(tmp_path):
    write_yaml(tmp_path, "reports/pilot/balanced_pilot_v1.cumulative648.yaml", {
        "complete_and_artifact_valid": True,
        "protected_test_used": False,
        "counts": {"expected": 648, "observed": 647, "usable": 647},
        "integrity": {
            "missing_episode_keys": ["missing"],
            "unexpected_episode_keys": [],
            "duplicate_episode_keys": [],
            "replacement_errors": [],
        },
    })
    assert (
        "balanced pilot evidence missing or invalid: "
        "reports/pilot/balanced_pilot_v1.cumulative648.yaml"
    ) in audit_completion(tmp_path)

import hashlib
from pathlib import Path

import yaml

from src.nonmodel_completion import audit_nonmodel

ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_CHECKS = ("protected_split_unassigned_and_unused", "no_model_or_confirmatory_artifact_created")


def test_real_workspace_audit_has_explicit_protected_and_human_boundaries():
    report = audit_nonmodel(ROOT)
    assert report["protected_outcomes_consulted"] is False
    freeze_path = ROOT / "configs/model_freeze.yaml"
    freeze = yaml.safe_load(freeze_path.read_text(encoding="utf-8"))
    assert freeze["frozen"] is True
    assert report["model_freeze"] == {
        "frozen": True, "path": "configs/model_freeze.yaml",
        "sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
        "frozen_utc": freeze["frozen_utc"],
    }
    # Post-freeze the pre-model boundaries are expected to be crossed: they are reported
    # as informational ``post_freeze`` values and never counted as incomplete work.
    assert report["post_freeze"]["status"] == "post_freeze"
    assert report["post_freeze"]["expected_value_after_freeze"] is False
    for name in BOUNDARY_CHECKS:
        assert name not in report["checks"]
        assert report["post_freeze"][name] is False
        assert name not in report["incomplete_checks"]
    assert report["external_human_requirements"]["first_reviewer_annotations"] == "20/20"
    assert report["external_human_requirements"][
        "second_distinct_reviewer_annotations"
    ].endswith("/20")  # optional second reviews accrue over time
    assert report["external_human_requirements"]["researcher_threshold_approval"] is True
    assert report["external_human_requirements"]["protocol_1_1_amendment_approved"] is True
    assert "validation_collection_324_complete" in report["checks"]


def test_pre_freeze_audit_counts_the_boundary_checks(tmp_path):
    report = audit_nonmodel(tmp_path)
    assert report["model_freeze"] == {"frozen": False, "path": None, "sha256": None, "frozen_utc": None}
    assert report["post_freeze"] == {}
    for name in BOUNDARY_CHECKS:
        assert report["checks"][name] is True
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/model_freeze.yaml").write_text("frozen: false\n", encoding="utf-8")
    report = audit_nonmodel(tmp_path)
    assert report["model_freeze"]["frozen"] is False
    assert report["checks"]["no_model_or_confirmatory_artifact_created"] is False
    assert "no_model_or_confirmatory_artifact_created" in report["incomplete_checks"]

from pathlib import Path

import pytest
import yaml

from src.review_app import ReviewStore
from scripts.manual_audit_app import valid_basic_auth


def write_yaml(root: Path, relative: str, value: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def make_store(tmp_path: Path) -> ReviewStore:
    taxonomy = {
        "fault_families": {"none": {}},
        "terminal_outcomes": {"navigation_abort": {}},
        "exclusion_reasons": {"infrastructure_failure": {"exclude_episode": True}},
    }
    annotation = {
        "schema_version": 1,
        "run_id": "run-1",
        "episode": {"start_time": 0.0, "end_time": 10.0, "termination_reason": "success"},
        "injections": [{"family": "none", "severity": None, "planned_onset": None,
                        "actual_onset": None, "duration_seconds": None, "eligible": False}],
        "events": [],
        "episode_exclusion": {"excluded": False, "reason": None},
        "review": {"second_reviewer": None, "adjudication_status": "pending",
                   "disagreements": []},
    }
    decisions = {
        "localisation_loss": {
            "decision": "accept", "translation_error_m": .5, "yaw_error_rad": .5,
            "persistence_seconds": 2.0,
            "threshold_logic": "either_translation_or_yaw_continuously",
            "rationale": "accepted",
        },
        "immobilisation": {
            "decision": "accept", "minimum_command_speed_mps": .05,
            "maximum_progress_m": .5, "persistence_seconds": 10.0,
            "angular_only_commands_count": False,
            "progress_measure": "planar_odometry_displacement", "rationale": "accepted",
        },
        "unsafe_perception": {
            "decision": "accept", "protected_stopping_region_m_from_footprint": .42,
            "added_braking_or_latency_margin_m": 0.0, "rationale": "accepted",
        },
    }
    template = {
        "schema_version": 1, "protocol_version": "1.0", "proposal_file": "proposal.yaml",
        "reviewer_name": "TODO", "reviewer_role": "TODO", "reviewed_utc": "TODO",
        "protected_outcomes_consulted": False,
        "decisions": {name: dict(value) for name, value in decisions.items()},
        "overall_decision": "pending", "requires_protocol_version_change": False,
        "requires_existing_development_data_invalidation": False,
    }
    write_yaml(tmp_path, "configs/failure_taxonomy.yaml", taxonomy)
    write_yaml(tmp_path, "configs/event_threshold_review.2026-08-30.researcher.yaml", {
        "reviewer_name": "Researcher One", "decisions": decisions,
    })
    write_yaml(tmp_path, "configs/event_threshold_review.template.yaml", template)
    write_yaml(tmp_path, "data/annotations/run-1.yaml", annotation)
    return ReviewStore(tmp_path)


def test_two_distinct_reviewers_create_gate_valid_copy(tmp_path):
    store = make_store(tmp_path)
    first = store.record_review(
        "run-1", reviewer_name="Reviewer One", decision="agree", notes="",
        evidence_checked=True,
    )
    assert first["status"] == "first_complete"
    first_copy = yaml.safe_load(
        (tmp_path / "data/annotations/reviewed/run-1.yaml").read_text(encoding="utf-8")
    )
    assert first_copy["review"]["first_review_notes"] is None
    with pytest.raises(ValueError, match="different person"):
        store.record_review(
            "run-1", reviewer_name="reviewer one", decision="agree", notes="",
            evidence_checked=True,
        )
    second = store.record_review(
        "run-1", reviewer_name="Reviewer Two", decision="agree", notes="",
        evidence_checked=True,
    )
    assert second["status"] == "complete"
    reviewed = yaml.safe_load(
        (tmp_path / "data/annotations/reviewed/run-1.yaml").read_text(encoding="utf-8")
    )
    assert reviewed["review"]["adjudication_status"] == "agreed"
    assert reviewed["review"]["second_review_notes"] is None


def test_disagreement_is_retained_without_changing_causal_fields(tmp_path):
    store = make_store(tmp_path)
    result = store.record_review(
        "run-1", reviewer_name="Reviewer One", decision="disagree",
        notes="Terminal evidence is ambiguous.", evidence_checked=True,
    )
    assert result["status"] == "flagged"
    assert len(list((tmp_path / "data/annotations/review-disagreements").glob("*.yaml"))) == 1
    reviewed = yaml.safe_load(
        (tmp_path / "data/annotations/reviewed/run-1.yaml").read_text(encoding="utf-8")
    )
    automatic = store.automatic("run-1")
    assert reviewed["episode"] == automatic["episode"]
    assert reviewed["events"] == automatic["events"]


def test_supervisor_signoff_requires_distinct_identity_and_is_not_overwritten(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        store.record_supervisor_review(
            reviewer_name="Researcher One", rationale="Methodologically acceptable.",
            role_confirmed=True, protected_confirmed=True,
        )
    result = store.record_supervisor_review(
        reviewer_name="Supervisor Two", rationale="Prospectively accepted without protected outcomes.",
        role_confirmed=True, protected_confirmed=True,
    )
    assert result["status"] == "complete"
    with pytest.raises(ValueError, match="refusing to overwrite"):
        store.record_supervisor_review(
            reviewer_name="Supervisor Three", rationale="Prospectively accepted again.",
            role_confirmed=True, protected_confirmed=True,
        )


def test_episode_state_uses_frozen_injection_severity_when_summary_is_null(tmp_path):
    store = make_store(tmp_path)
    annotation_path = tmp_path / "data/annotations/run-1.yaml"
    annotation = yaml.safe_load(annotation_path.read_text(encoding="utf-8"))
    annotation["injections"][0].update({"family": "camera_occlusion", "severity": "high"})
    write_yaml(tmp_path, "data/annotations/run-1.yaml", annotation)
    write_yaml(tmp_path, "data/raw/summaries/run-1.yaml", {
        "identity": {"run_id": "run-1", "episode_key": "camera-high"},
        "label_only": {"fault_family": "camera_occlusion", "fault_severity": None},
        "outcome": {"terminal_state": "success"},
    })
    store = ReviewStore(tmp_path)
    episode = store.episode_state()[0]
    assert episode["fault_family"] == "camera_occlusion"
    assert episode["fault_severity"] == "high"


def test_episode_state_ignores_unrelated_incomplete_campaign_summary(tmp_path):
    store = make_store(tmp_path)
    incomplete = tmp_path / "data/raw/summaries/current-campaign-run.yaml"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_bytes(b"")
    episode = store.episode_state()[0]
    assert episode["run_id"] == "run-1"


def test_review_app_basic_auth_is_fail_closed_when_password_is_set():
    import base64

    valid = "Basic " + base64.b64encode(b"reviewer:secret").decode("ascii")
    wrong = "Basic " + base64.b64encode(b"reviewer:wrong").decode("ascii")
    assert valid_basic_auth(valid, "reviewer", "secret") is True
    assert valid_basic_auth(wrong, "reviewer", "secret") is False
    assert valid_basic_auth(None, "reviewer", "secret") is False
    assert valid_basic_auth(None, "reviewer", "") is True

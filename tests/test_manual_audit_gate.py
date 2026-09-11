from src.labels.audit_gate import (
    causal_signature,
    validate_primary_review,
    validate_review_pair,
)
from scripts.check_manual_audit_gate import campaign_summaries


TAXONOMY = {
    "fault_families": {"none": {}},
    "terminal_outcomes": {"navigation_abort": {}},
    "exclusion_reasons": {"infrastructure_failure": {"exclude_episode": True}},
}


def annotation():
    return {
        "run_id": "run-1",
        "episode": {"start_time": 1.0, "end_time": 10.0, "termination_reason": "success"},
        "injections": [{"family": "none", "severity": None, "planned_onset": None,
                        "actual_onset": None, "duration_seconds": None, "eligible": False}],
        "events": [],
        "episode_exclusion": {"excluded": False, "reason": None},
        "review": {"first_reviewer": "A", "second_reviewer": "B",
                   "adjudication_status": "agreed"},
    }


def test_exact_independent_agreement_passes():
    automatic = annotation()
    reviewed = annotation()
    assert validate_review_pair(automatic, reviewed, TAXONOMY) == []


def test_exact_single_reviewer_agreement_passes_primary_gate():
    automatic = annotation()
    reviewed = annotation()
    reviewed["review"] = {
        "first_reviewer": "Researcher",
        "first_reviewed_utc": "2026-08-31T22:11:37Z",
        "second_reviewer": None,
        "adjudication_status": "pending",
    }
    assert validate_primary_review(automatic, reviewed, TAXONOMY) == []


def test_primary_gate_still_rejects_causal_changes():
    automatic = annotation()
    reviewed = annotation()
    reviewed["review"]["first_reviewed_utc"] = "2026-08-31T22:11:37Z"
    reviewed["episode"]["end_time"] = 9.5
    errors = validate_primary_review(automatic, reviewed, TAXONOMY)
    assert any("exactly match" in error for error in errors)


def test_causal_change_and_same_reviewer_fail_gate():
    automatic = annotation()
    reviewed = annotation()
    reviewed["episode"]["end_time"] = 9.5
    reviewed["review"]["second_reviewer"] = "A"
    errors = validate_review_pair(automatic, reviewed, TAXONOMY)
    assert any("different people" in error for error in errors)
    assert any("exactly match" in error for error in errors)
    assert causal_signature(automatic) != causal_signature(reviewed)


def test_campaign_summary_scan_ignores_retained_empty_payload(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_bytes(b"")
    valid = tmp_path / "valid.yaml"
    valid.write_text(
        "identity:\n  campaign_id: audit\n  episode_key: episode-1\n",
        encoding="utf-8",
    )
    assert list(campaign_summaries([empty, valid], "audit")) == ["episode-1"]

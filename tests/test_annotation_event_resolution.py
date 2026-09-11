import pytest

from scripts.extract_episode_annotation import resolve_primary_event


def summary():
    return {
        "identity": {"run_id": "run"},
        "environment": {"protected_test_used": False},
        "label_only": {"primary_event_class": "navigation_abort"},
    }


def test_operational_candidate_precedes_terminal_summary_event():
    operational = {
        "run_id": "run", "protected_test_used": False,
        "status": "automatic_review_pending",
        "primary_candidate": {"class": "localisation_loss", "time": 8.0},
    }
    assert resolve_primary_event(summary(), {"time": 10.0}, operational) \
        == ("localisation_loss", 8.0)


def test_terminal_fallback_remains_available_for_audit_compatibility():
    assert resolve_primary_event(summary(), {"time": 10.0}, None) \
        == ("navigation_abort", 10.0)


def test_operational_candidate_must_match_run_and_precede_terminal():
    operational = {
        "run_id": "other", "protected_test_used": False,
        "status": "automatic_review_pending", "primary_candidate": None,
    }
    with pytest.raises(ValueError, match="run_id differs"):
        resolve_primary_event(summary(), {"time": 10.0}, operational)
    operational["run_id"] = "run"
    operational["primary_candidate"] = {"class": "immobilisation", "time": 11.0}
    with pytest.raises(ValueError, match="after terminal"):
        resolve_primary_event(summary(), {"time": 10.0}, operational)

from src.recovery import GuardConfig
from src.recovery.verification import verify_guard_space


def test_exhaustive_discrete_guard_space_has_no_selector_bypass():
    report = verify_guard_space(GuardConfig())
    assert report["states_checked"] > 1_000
    assert report["policy_decisions_checked"] > 10_000
    assert report["violation_count"] == 0
    assert report["passed"] is True
    assert report["eligibility_counts"]["request_assistance"] == report["states_checked"]

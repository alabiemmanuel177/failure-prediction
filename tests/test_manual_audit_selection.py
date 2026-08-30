from pathlib import Path

from scripts.prepare_manual_audit import select_audit_keys


def summary(state: str = "success") -> tuple[Path, dict]:
    return Path("summary.yaml"), {"outcome": {"terminal_state": state}}


def test_pre_summary_infrastructure_failure_uses_first_frozen_reserve():
    primary = [f"p{index}" for index in range(20)]
    reserves = [f"r{index}" for index in range(5)]
    by_key = {key: summary() for key in primary[1:] + reserves}
    selected = select_audit_keys(primary, reserves, by_key, {"p0"})
    assert selected == primary[1:] + ["r0"]


def test_artifact_invalid_summary_also_uses_frozen_reserve():
    primary = [f"p{index}" for index in range(20)]
    reserves = [f"r{index}" for index in range(5)]
    by_key = {key: summary() for key in primary + reserves}
    selected = select_audit_keys(primary, reserves, by_key, {"p0"})
    assert selected == primary[1:] + ["r0"]


def test_invalid_reserve_is_skipped_without_reordering_remaining_reserves():
    primary = [f"p{index}" for index in range(20)]
    reserves = [f"r{index}" for index in range(5)]
    by_key = {key: summary() for key in primary + reserves}
    selected = select_audit_keys(primary, reserves, by_key, {"p0", "r0"})
    assert selected == primary[1:] + ["r1"]

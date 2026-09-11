import pytest

from src.protected_data import enforce_protected_boundary


def test_preprotected_access_requires_explicit_false_marker():
    enforce_protected_boundary(
        False, explicitly_allowed=False, confirmatory_gate_passed=False
    )
    with pytest.raises(ValueError, match="explicitly declare"):
        enforce_protected_boundary(
            None, explicitly_allowed=False, confirmatory_gate_passed=False
        )


def test_protected_access_requires_both_keys():
    with pytest.raises(ValueError, match="explicit approval"):
        enforce_protected_boundary(
            True, explicitly_allowed=False, confirmatory_gate_passed=True
        )
    with pytest.raises(ValueError, match="before confirmatory freeze"):
        enforce_protected_boundary(
            True, explicitly_allowed=True, confirmatory_gate_passed=False
        )
    enforce_protected_boundary(
        True, explicitly_allowed=True, confirmatory_gate_passed=True
    )

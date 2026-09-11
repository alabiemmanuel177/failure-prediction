import numpy as np
import pytest

from scripts.assemble_episode_sequences import publish_npz
from src.protected_data import enforce_protected_boundary


def test_sequence_npz_publication_is_atomic_and_immutable(tmp_path):
    target = tmp_path / "nested" / "sequence.npz"
    publish_npz(target, {"X": np.zeros((1, 2, 3), dtype=np.float32)})
    with np.load(target, allow_pickle=False) as artifact:
        assert artifact["X"].shape == (1, 2, 3)
    assert list(target.parent.glob(".*.tmp")) == []
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_npz(target, {"X": np.ones((1, 2, 3), dtype=np.float32)})


def test_protected_sequence_extraction_requires_explicit_post_freeze_gate():
    enforce_protected_boundary(
        False, explicitly_allowed=False, confirmatory_gate_passed=False
    )
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
    with pytest.raises(ValueError, match="explicitly declare"):
        enforce_protected_boundary(
            None, explicitly_allowed=False, confirmatory_gate_passed=False
        )

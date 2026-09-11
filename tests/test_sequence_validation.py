import json

import numpy as np

from scripts.assemble_episode_sequences import publish_npz
from src.features import validate_sequence_artifact


def arrays(*, protected=False, names=("x", "x__age_seconds", "x__missing")):
    metadata = {
        "sequence_count": 2,
        "time_steps": 4,
        "feature_columns": list(names),
        "protected_test_used": protected,
        **{name: "a" * 64 for name in (
            "feature_schema_sha256", "leakage_denylist_sha256", "telemetry_sha256",
            "labels_sha256", "annotation_sha256", "summary_sha256",
        )},
    }
    return {
        "X": np.zeros((2, 4, len(names)), dtype=np.float32),
        "y": np.asarray([0, 1], dtype=np.int8),
        "decision_index": np.asarray([1, 2], dtype=np.int64),
        "decision_time": np.asarray([5.0, 5.5], dtype=np.float64),
        "feature_names": np.asarray(names),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }


def test_sequence_validator_accepts_exact_unprotected_contract(tmp_path):
    path = tmp_path / "valid.npz"
    names = ("x", "x__age_seconds", "x__missing")
    publish_npz(path, arrays(names=names))
    assert validate_sequence_artifact(path, expected_feature_names=names) == []


def test_sequence_validator_rejects_reordered_columns_bad_labels_and_protected(tmp_path):
    path = tmp_path / "invalid.npz"
    payload = arrays(protected=True)
    payload["y"] = np.asarray([0, 2], dtype=np.int8)
    publish_npz(path, payload)
    findings = validate_sequence_artifact(
        path, expected_feature_names=("x__missing", "x__age_seconds", "x")
    )
    assert "feature_names differ from the frozen model-column order" in findings
    assert "y contains a non-binary label" in findings
    assert "protected or unmarked sequence artifact is not allowed" in findings


def test_sequence_validator_accepts_explicit_zero_window_episode(tmp_path):
    path = tmp_path / "zero.npz"
    names = ("x", "x__age_seconds", "x__missing")
    payload = arrays(names=names)
    payload["X"] = np.empty((0, 4, len(names)), dtype=np.float32)
    payload["y"] = np.empty((0,), dtype=np.int8)
    payload["decision_index"] = np.empty((0,), dtype=np.int64)
    payload["decision_time"] = np.empty((0,), dtype=np.float64)
    metadata = json.loads(str(payload["metadata_json"]))
    metadata["sequence_count"] = 0
    payload["metadata_json"] = np.asarray(json.dumps(metadata))
    publish_npz(path, payload)
    assert validate_sequence_artifact(path, expected_feature_names=names) == []

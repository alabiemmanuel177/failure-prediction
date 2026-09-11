"""Tests for the all-decisions artifact contract and the derived-dataset drivers."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    FeatureSpec, LeakagePolicy, ScalarSample, assemble_causal_sequences, model_columns,
    validate_decision_artifact,
)


PRIMARY = ("valid_return_fraction",)


def _labels(start: float, end: float):
    rows = []
    index = 0
    t = start + 5.0
    while t <= end + 1e-9:
        eligibility = "eligible_negative" if t < 12 else "excluded_too_late" if t < 13 else "eligible_positive"
        rows.append({
            "decision_index": index, "decision_time": round(t, 3),
            "label": {"eligible_negative": 0, "eligible_positive": 1}.get(eligibility),
            "eligibility": eligibility,
        })
        index += 1
        t += 0.5
    return rows


def _examples(include_ineligible: bool):
    samples = [ScalarSample(t / 10, 0.9) for t in range(0, 200)]
    return assemble_causal_sequences(
        run_id="r", episode_start=0.0, episode_end=15.0, labels=_labels(0.0, 15.0),
        specs=[FeatureSpec("valid_return_fraction", "/scan", 0.5)],
        samples_by_feature={"valid_return_fraction": samples},
        leakage_policy=LeakagePolicy.from_yaml(ROOT / "configs/leakage_denylist.yaml"),
        primary_features=PRIMARY, goal_x=1.0, goal_y=1.0,
        include_ineligible=include_ineligible,
    )


def test_include_ineligible_keeps_every_decision_and_marks_excluded_as_minus_one():
    eligible = _examples(False)
    everything = _examples(True)
    assert len(everything) == len(_labels(0.0, 15.0))
    assert len(eligible) < len(everything)
    assert {example.label for example in everything} == {-1, 0, 1}
    by_index = {example.decision_index: example for example in everything}
    for example in eligible:
        assert by_index[example.decision_index].values == example.values
        assert by_index[example.decision_index].label == example.label


def _write_decision_artifact(path: Path, *, y, eligibility, protected=False, kind="all_decisions"):
    columns = model_columns(PRIMARY)
    count = len(y)
    metadata = {
        "artifact_kind": kind, "sequence_count": count, "time_steps": 10,
        "feature_columns": list(columns), "protected_test_used": protected,
    }
    np.savez(path, X=np.zeros((count, 10, len(columns)), dtype=np.float32),
             y=np.asarray(y, dtype=np.int8), eligibility=np.asarray(eligibility),
             decision_index=np.arange(count), decision_time=np.arange(count) * 0.5 + 5.0,
             feature_names=np.asarray(columns),
             metadata_json=np.asarray(json.dumps(metadata)))


def test_decision_artifact_validator_accepts_consistent_file(tmp_path):
    path = tmp_path / "d.npz"
    _write_decision_artifact(path, y=[0, -1, 1], eligibility=[
        "eligible_negative", "excluded_too_late", "eligible_positive"])
    assert validate_decision_artifact(path, expected_feature_names=model_columns(PRIMARY)) == []


@pytest.mark.parametrize("y,eligibility,expected", [
    ([0, 0], ["eligible_negative", "eligible_positive"], "eligibility and label disagree"),
    ([0, 2], ["eligible_negative", "eligible_positive"], "outside {-1, 0, 1}"),
])
def test_decision_artifact_validator_rejects_inconsistency(tmp_path, y, eligibility, expected):
    path = tmp_path / "d.npz"
    _write_decision_artifact(path, y=y, eligibility=eligibility)
    findings = validate_decision_artifact(path, expected_feature_names=model_columns(PRIMARY))
    assert any(expected in finding for finding in findings)


def test_decision_artifact_validator_refuses_protected_without_permission(tmp_path):
    path = tmp_path / "d.npz"
    _write_decision_artifact(path, y=[0], eligibility=["eligible_negative"], protected=True)
    findings = validate_decision_artifact(path, expected_feature_names=model_columns(PRIMARY))
    assert any("protected" in finding for finding in findings)
    assert validate_decision_artifact(
        path, expected_feature_names=model_columns(PRIMARY), allow_protected=True) == []


def test_extract_driver_refuses_protected_inventory(tmp_path):
    import subprocess
    inventory = tmp_path / "x.episodes.jsonl"
    inventory.write_text(json.dumps({"run_id": "a", "split": "development",
                                     "protected_test_used": True}) + "\n")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/extract_dataset_sequences.py"),
         "--inventory", str(inventory), "--dataset-id", "x", "--output-root", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "protected" in result.stderr + result.stdout

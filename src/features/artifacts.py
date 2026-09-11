"""Fail-closed validation for immutable per-episode sequence artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


REQUIRED_ARRAYS = {
    "X", "y", "decision_index", "decision_time", "feature_names", "metadata_json",
}


def validate_sequence_artifact(
    path: Path, *, expected_feature_names: Sequence[str],
    allow_protected: bool = False,
) -> list[str]:
    findings: list[str] = []
    try:
        artifact = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        return [f"cannot open sequence artifact: {error}"]
    with artifact:
        missing = sorted(REQUIRED_ARRAYS - set(artifact.files))
        if missing:
            return [f"sequence artifact lacks arrays: {missing}"]
        try:
            x = artifact["X"]
            y = artifact["y"]
            decision_index = artifact["decision_index"]
            decision_time = artifact["decision_time"]
            feature_names = tuple(str(value) for value in artifact["feature_names"].tolist())
            metadata = json.loads(str(artifact["metadata_json"].item()))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return [f"sequence artifact arrays cannot be decoded: {error}"]

        if x.ndim != 3:
            findings.append("X must have shape [examples,time,features]")
        count = x.shape[0] if x.ndim >= 1 else -1
        if y.shape != (count,) or decision_index.shape != (count,) \
                or decision_time.shape != (count,):
            findings.append("label and decision arrays do not match X example count")
        expected = tuple(expected_feature_names)
        if feature_names != expected:
            findings.append("feature_names differ from the frozen model-column order")
        if x.ndim == 3 and x.shape[2] != len(expected):
            findings.append("X feature width differs from frozen model columns")
        if not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
            findings.append("X must contain only finite floating-point values")
        if not np.isin(y, (0, 1)).all():
            findings.append("y contains a non-binary label")
        if count and (
            len(np.unique(decision_index)) != count
            or np.any(np.diff(decision_time.astype(float)) <= 0)
        ):
            findings.append("decision identities are duplicated or not strictly ordered")
        if not isinstance(metadata, dict):
            findings.append("metadata_json must decode to an object")
        else:
            if metadata.get("sequence_count") != count:
                findings.append("metadata sequence_count differs from X")
            if x.ndim == 3 and metadata.get("time_steps") != x.shape[1]:
                findings.append("metadata time_steps differs from X")
            if metadata.get("feature_columns") != list(expected):
                findings.append("metadata feature_columns differ from frozen order")
            protected = metadata.get("protected_test_used")
            if protected is not False and not (allow_protected and protected is True):
                findings.append("protected or unmarked sequence artifact is not allowed")
            for name in (
                "feature_schema_sha256", "leakage_denylist_sha256", "telemetry_sha256",
                "labels_sha256", "annotation_sha256", "summary_sha256",
            ):
                value = metadata.get(name)
                if not isinstance(value, str) or len(value) != 64 \
                        or any(char not in "0123456789abcdef" for char in value):
                    findings.append(f"metadata {name} is not a SHA-256 digest")
    return findings


def validate_decision_artifact(
    path: Path, *, expected_feature_names: Sequence[str],
    allow_protected: bool = False,
) -> list[str]:
    """Validate an all-decisions artifact (labels in {-1, 0, 1} plus eligibility)."""
    findings: list[str] = []
    try:
        artifact = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        return [f"cannot open decision artifact: {error}"]
    with artifact:
        missing = sorted((REQUIRED_ARRAYS | {"eligibility"}) - set(artifact.files))
        if missing:
            return [f"decision artifact lacks arrays: {missing}"]
        try:
            x = artifact["X"]
            y = artifact["y"]
            eligibility = [str(value) for value in artifact["eligibility"].tolist()]
            decision_index = artifact["decision_index"]
            decision_time = artifact["decision_time"]
            feature_names = tuple(str(value) for value in artifact["feature_names"].tolist())
            metadata = json.loads(str(artifact["metadata_json"].item()))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return [f"decision artifact arrays cannot be decoded: {error}"]
        count = x.shape[0] if x.ndim >= 1 else -1
        if x.ndim != 3:
            findings.append("X must have shape [decisions,time,features]")
        if y.shape != (count,) or decision_index.shape != (count,) \
                or decision_time.shape != (count,) or len(eligibility) != count:
            findings.append("label, eligibility and decision arrays do not match X")
        expected = tuple(expected_feature_names)
        if feature_names != expected:
            findings.append("feature_names differ from the frozen model-column order")
        if not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
            findings.append("X must contain only finite floating-point values")
        if not np.isin(y, (-1, 0, 1)).all():
            findings.append("y contains a value outside {-1, 0, 1}")
        for label, name in zip(y.tolist(), eligibility):
            expected_label = {"eligible_positive": 1, "eligible_negative": 0}.get(name, -1)
            if label != expected_label:
                findings.append("eligibility and label disagree")
                break
        if count and np.any(np.diff(decision_index.astype(int)) != 1):
            findings.append("decision indices are not consecutive")
        if count and np.any(np.diff(decision_time.astype(float)) <= 0):
            findings.append("decision times are not strictly ordered")
        if not isinstance(metadata, dict):
            findings.append("metadata_json must decode to an object")
        else:
            if metadata.get("artifact_kind") != "all_decisions":
                findings.append("metadata artifact_kind is not all_decisions")
            if metadata.get("sequence_count") != count:
                findings.append("metadata sequence_count differs from X")
            if metadata.get("feature_columns") != list(expected):
                findings.append("metadata feature_columns differ from frozen order")
            protected = metadata.get("protected_test_used")
            if protected is not False and not (allow_protected and protected is True):
                findings.append("protected or unmarked decision artifact is not allowed")
    return findings

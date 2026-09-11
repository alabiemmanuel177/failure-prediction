"""Synthetic derived-dataset fixtures for the predictor tests (never real data).

Writes ``<root>/<dataset_id>/extraction_manifest.jsonl`` plus ``sequences/*.npz``
with exactly the keys ``scripts/assemble_episode_sequences.py`` writes, and
``decisions/*.npz`` with the additional ``eligibility`` array and ``y`` in
{-1, 0, 1}, so the trainer and predictor are exercised on the contract layout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.models.common import frozen_feature_names


FEATURES = frozen_feature_names()
STEPS = 10
STRIDE = 0.5
FAMILIES = ("none", "lidar_dropout", "wheel_slip", "dynamic_blockage")


def _metadata(run_id: str, split: str, map_id: str, count: int) -> str:
    digest = hashlib.sha256(run_id.encode()).hexdigest()
    return json.dumps({
        "schema_version": 1, "run_id": run_id, "split": split, "map_id": map_id,
        "route_id": f"{map_id}_r0", "feature_schema_sha256": digest,
        "leakage_denylist_sha256": digest, "telemetry_sha256": digest, "labels_sha256": digest,
        "annotation_sha256": digest, "summary_sha256": digest, "history_seconds": 5.0,
        "stride_seconds": STRIDE, "sequence_count": count, "time_steps": STEPS,
        "feature_columns": list(FEATURES), "protected_test_used": False,
    }, sort_keys=True, separators=(",", ":"))


def _episode_arrays(rng: np.random.Generator, run_id: str, has_event: bool, decisions: int):
    """Decision grid with a clear precursor in pose_covariance_trace before the event."""
    channels = len(FEATURES)
    index = {name: i for i, name in enumerate(FEATURES)}
    X = rng.standard_normal((decisions, STEPS, channels)).astype(np.float32) * 0.3
    for name, position in index.items():
        if name.endswith("__missing"):
            X[..., position] = (rng.random((decisions, STEPS)) < 0.03).astype(np.float32)
        elif name.endswith("__age_seconds"):
            X[..., position] = np.abs(X[..., position]) * 0.1
    decision_time = 12.5 + STRIDE * np.arange(decisions)
    eligibility = np.full(decisions, "eligible_negative", dtype="<U40")
    event_time = None
    if has_event:
        event_time = float(decision_time[-1] + 1.0)
        positive = decision_time >= event_time - 10.0
        too_late = decision_time > event_time - 1.0
        near = (decision_time >= event_time - 30.0) & ~positive
        eligibility[near] = "excluded_near_event_or_injection"
        eligibility[positive] = "eligible_positive"
        eligibility[too_late] = "excluded_too_late"
        ramp = np.linspace(1.0, 4.0, STEPS, dtype=np.float32)
        X[positive, :, index["pose_covariance_trace"]] += ramp
        X[positive, :, index["progress_slope"]] -= 1.5
    y = np.where(eligibility == "eligible_positive", 1,
                 np.where(eligibility == "eligible_negative", 0, -1)).astype(np.int8)
    return X, y, decision_time, eligibility, event_time


def _publish(path: Path, arrays: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset(
    root: Path, dataset_id: str, split: str, *, episodes: int = 8, seed: int = 7,
    decisions: int = 60, protected: bool = False, write_decisions: bool = True,
    map_prefix: str | None = None,
) -> Path:
    rng = np.random.default_rng(seed)
    dataset_root = root / dataset_id
    rows = []
    prefix = map_prefix or ("dev" if split == "development" else "val")
    for number in range(episodes):
        family = FAMILIES[number % len(FAMILIES)]
        has_event = family != "none"
        run_id = f"{dataset_id}-ep{number:02d}"
        map_id = f"{prefix}_{number % 2:02d}"
        X, y, decision_time, eligibility, event_time = _episode_arrays(rng, run_id, has_event, decisions)
        decision_index = np.arange(decisions, dtype=np.int64)
        eligible = y >= 0
        metadata = _metadata(run_id, split, map_id, int(eligible.sum()))
        if protected:
            metadata = metadata.replace('"protected_test_used":false', '"protected_test_used":true')
        sequence_sha = _publish(dataset_root / "sequences" / f"{run_id}.npz", {
            "X": X[eligible], "y": y[eligible], "decision_index": decision_index[eligible],
            "decision_time": decision_time[eligible], "feature_names": np.asarray(FEATURES),
            "metadata_json": np.asarray(metadata),
        })
        if write_decisions:
            _publish(dataset_root / "decisions" / f"{run_id}.npz", {
                "X": X, "y": y, "decision_index": decision_index, "decision_time": decision_time,
                "eligibility": eligibility, "feature_names": np.asarray(FEATURES),
                "metadata_json": np.asarray(metadata.replace(
                    f'"sequence_count":{int(eligible.sum())}', f'"sequence_count":{decisions}')),
            })
        rows.append({
            "run_id": run_id, "split": split, "map_id": map_id, "route_id": f"{map_id}_r0",
            "fault_family": family, "severity": "medium" if has_event else "none",
            "seed": 1000 + number, "protected_test_used": protected,
            "artifact_sha256": {"sequences": sequence_sha},
            "primary_event_class": "navigation_abort" if has_event else None,
            "primary_event_time": event_time,
            "positive_sequences": int((y == 1).sum()), "negative_sequences": int((y == 0).sum()),
        })
    manifest = dataset_root / "extraction_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return manifest

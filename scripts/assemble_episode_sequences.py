#!/usr/bin/env python3
"""Assemble one episode's eligible, fixed-rate causal sequences into immutable NPZ."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    LeakagePolicy, ScalarSample, assemble_causal_sequences,
    load_primary_feature_set, load_raw_feature_contract, model_columns,
    resolve_feature_specs,
)
from src.protected_data import enforce_protected_boundary  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_npz(path: Path, arrays: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-protected-after-freeze", action="store_true")
    args = parser.parse_args()

    telemetry = list(csv.DictReader(args.telemetry.open(newline="", encoding="utf-8")))
    labels = list(csv.DictReader(args.labels.open(newline="", encoding="utf-8")))
    annotation = yaml.safe_load(args.annotation.read_text(encoding="utf-8"))
    summary = yaml.safe_load(args.summary.read_text(encoding="utf-8"))
    if not isinstance(annotation, dict) or not isinstance(summary, dict):
        raise SystemExit("annotation and summary must be YAML mappings")
    protected = summary.get("environment", {}).get("protected_test_used")
    confirmatory_gate_passed = False
    if protected is True and args.allow_protected_after_freeze:
        confirmatory_gate_passed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_readiness.py"),
             "--stage", "confirmatory"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
    try:
        enforce_protected_boundary(
            protected,
            explicitly_allowed=args.allow_protected_after_freeze,
            confirmatory_gate_passed=confirmatory_gate_passed,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    run_id = str(summary["identity"]["run_id"])
    if annotation.get("run_id") != run_id or any(row.get("run_id") != run_id for row in telemetry + labels):
        raise SystemExit("telemetry, labels, annotation, and summary run_id must match")

    schema_path = ROOT / "configs/feature_schema.yaml"
    contract = load_raw_feature_contract(schema_path)
    specs, grouped = resolve_feature_specs(contract, telemetry)
    samples = {
        name: [ScalarSample(float(row["timestamp"]), float(row["value"])) for row in rows]
        for name, rows in grouped.items()
    }
    for row in labels:
        raw_label = row.get("label")
        row["label"] = int(raw_label) if raw_label in {"0", "1"} else None
        row["decision_index"] = int(row["decision_index"])
        row["decision_time"] = float(row["decision_time"])
    goal = summary.get("environment", {}).get("goal_pose")
    if not isinstance(goal, dict):
        raise SystemExit("summary lacks goal_pose")
    windowing = yaml.safe_load(
        (ROOT / "configs/failure_events.yaml").read_text(encoding="utf-8")
    )["windowing"]
    primary = load_primary_feature_set(schema_path)
    examples = assemble_causal_sequences(
        run_id=run_id,
        episode_start=float(annotation["episode"]["start_time"]),
        episode_end=float(annotation["episode"]["end_time"]),
        labels=labels,
        specs=specs,
        samples_by_feature=samples,
        leakage_policy=LeakagePolicy.from_yaml(ROOT / "configs/leakage_denylist.yaml"),
        primary_features=primary,
        goal_x=float(goal["x"]),
        goal_y=float(goal["y"]),
        history_seconds=float(windowing["history_seconds"]),
        stride_seconds=float(windowing["decision_stride_seconds"]),
    )
    columns = model_columns(primary)
    time_steps = round(
        float(windowing["history_seconds"]) / float(windowing["decision_stride_seconds"])
    )
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "split": summary["environment"].get("split"),
        "map_id": summary["environment"].get("map_id"),
        "route_id": summary["environment"].get("route_id"),
        "feature_schema_sha256": sha256(schema_path),
        "leakage_denylist_sha256": sha256(ROOT / "configs/leakage_denylist.yaml"),
        "telemetry_sha256": sha256(args.telemetry),
        "labels_sha256": sha256(args.labels),
        "annotation_sha256": sha256(args.annotation),
        "summary_sha256": sha256(args.summary),
        "history_seconds": float(windowing["history_seconds"]),
        "stride_seconds": float(windowing["decision_stride_seconds"]),
        "sequence_count": len(examples),
        "time_steps": time_steps,
        "feature_columns": list(columns),
        "protected_test_used": protected,
    }
    publish_npz(args.output, {
        "X": (
            np.asarray([example.values for example in examples], dtype=np.float32)
            if examples else np.empty((0, time_steps, len(columns)), dtype=np.float32)
        ),
        "y": np.asarray([example.label for example in examples], dtype=np.int8),
        "decision_index": np.asarray([example.decision_index for example in examples], dtype=np.int64),
        "decision_time": np.asarray([example.decision_time for example in examples], dtype=np.float64),
        "feature_names": np.asarray(columns),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    })
    print(json.dumps({
        "output": str(args.output), "run_id": run_id,
        "shape": [len(examples), time_steps, len(columns)],
        "positive": sum(example.label for example in examples),
        "negative": sum(example.label == 0 for example in examples),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

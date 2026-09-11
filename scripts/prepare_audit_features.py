#!/usr/bin/env python3
"""Prepare non-training labels and causal features for the frozen audit sample."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    annotations = sorted(
        path for path in (ROOT / "data/annotations").glob("*.yaml")
        if path.name != "episode.template.yaml"
    )
    if len(annotations) != 20:
        raise SystemExit(f"expected frozen 20-episode audit sample, found {len(annotations)}")
    summaries = {}
    for path in (ROOT / "data/raw/summaries").glob("*.yaml"):
        summary = yaml.safe_load(path.read_text(encoding="utf-8"))
        summaries[summary["identity"]["run_id"]] = summary
    roots = {
        "labels": ROOT / "data/audit/automatic_labels",
        "telemetry": ROOT / "data/audit/scalar_telemetry",
        "features": ROOT / "data/audit/causal_features",
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    completed = 0
    for annotation_path in annotations:
        annotation = yaml.safe_load(annotation_path.read_text(encoding="utf-8"))
        run_id = annotation["run_id"]
        if run_id not in summaries:
            raise SystemExit(f"summary missing for audit run {run_id}")
        outputs = {name: root / f"{run_id}.csv" for name, root in roots.items()}
        if any(path.exists() for path in outputs.values()):
            if args.resume and all(path.exists() for path in outputs.values()):
                completed += 1
                continue
            raise SystemExit(f"partial or existing audit outputs for {run_id}; refusing overwrite")
        subprocess.run([
            sys.executable, str(ROOT / "scripts/generate_labels.py"),
            str(annotation_path), str(outputs["labels"]),
        ], check=True)
        subprocess.run([
            sys.executable, str(ROOT / "scripts/extract_bag_scalar_telemetry.py"),
            summaries[run_id]["provenance"]["bag_path"], run_id,
            str(outputs["telemetry"]),
        ], check=True)
        subprocess.run([
            sys.executable, str(ROOT / "scripts/extract_scalar_features.py"),
            str(outputs["telemetry"]), str(outputs["labels"]),
            str(outputs["features"]),
        ], check=True)
        completed += 1
    print(f"prepared audit-only causal labels and features for {completed}/20 episodes")
    print("admission: forbidden for model training until check_manual_audit_gate.py passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

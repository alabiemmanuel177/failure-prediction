#!/usr/bin/env python3
"""Derive review-pending temporal event evidence for the frozen 20-episode audit."""

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
    summaries: dict[str, Path] = {}
    for path in (ROOT / "data/raw/summaries").glob("*.yaml"):
        summary = yaml.safe_load(path.read_text(encoding="utf-8"))
        summaries[str(summary["identity"]["run_id"])] = path
    output_root = ROOT / "reports/manual-audit/operational-events"
    completed = 0
    for annotation_path in annotations:
        annotation = yaml.safe_load(annotation_path.read_text(encoding="utf-8"))
        run_id = str(annotation["run_id"])
        if run_id not in summaries:
            raise SystemExit(f"summary missing for audit run {run_id}")
        output = output_root / f"{run_id}.yaml"
        if output.exists():
            if args.resume:
                completed += 1
                continue
            raise SystemExit(f"refusing existing output {output}")
        subprocess.run([
            sys.executable, str(ROOT / "scripts/derive_operational_events.py"),
            str(summaries[run_id]), str(output),
        ], check=True)
        completed += 1
    print(f"prepared review-pending operational event evidence for {completed}/20 episodes")
    print("admission: human timeline confirmation remains mandatory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

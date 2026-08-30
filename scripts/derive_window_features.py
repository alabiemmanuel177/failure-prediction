#!/usr/bin/env python3
"""Enrich synchronized scalar rows with causal five-second temporal features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import derive_window_features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.features.open(newline="", encoding="utf-8")))
    summary = yaml.safe_load(args.summary.read_text(encoding="utf-8"))
    run_ids = {row["run_id"] for row in rows}
    if run_ids != {summary["identity"]["run_id"]}:
        raise SystemExit("feature and summary run_id do not match")
    goal = summary["environment"].get("goal_pose")
    if not goal:
        raise SystemExit("summary lacks goal_pose; do not infer it from protected outcomes")
    enriched = derive_window_features(
        rows, goal_x=float(goal["x"]), goal_y=float(goal["y"]), history_seconds=5.0
    )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(enriched[0].keys()))
        writer.writeheader(); writer.writerows(enriched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

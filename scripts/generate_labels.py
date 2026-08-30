#!/usr/bin/env python3
"""Generate an auditable causal label CSV from one episode annotation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labels import LabelConfig, label_decision_times
from src.labels.annotations import validate_annotation


def load_yaml(path: Path):
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rules = load_yaml(ROOT / "configs" / "failure_events.yaml")
    annotation = load_yaml(args.annotation)
    taxonomy = load_yaml(ROOT / "configs" / "failure_taxonomy.yaml")
    annotation_errors = validate_annotation(annotation, taxonomy)
    if annotation_errors:
        details = "\n".join(f"- {error}" for error in annotation_errors)
        raise SystemExit(f"Refusing invalid annotation:\n{details}")
    if annotation.get("episode_exclusion", {}).get("excluded"):
        raise SystemExit("Refusing to label an episode marked excluded")

    windows = rules["windowing"]
    config = LabelConfig(
        history_seconds=float(windows["history_seconds"]),
        warning_horizon_seconds=float(windows["warning_horizon_seconds"]),
        too_late_guard_seconds=float(windows["too_late_guard_seconds"]),
        negative_guard_seconds=float(windows["negative_guard_seconds"]),
        decision_stride_seconds=float(windows["decision_stride_seconds"]),
    )
    onsets = [
        injection["actual_onset"]
        for injection in annotation.get("injections", [])
        if injection.get("eligible") is True and injection.get("actual_onset") is not None
    ]
    records = label_decision_times(
        episode_start=float(annotation["episode"]["start_time"]),
        episode_end=float(annotation["episode"]["end_time"]),
        events=annotation.get("events", []),
        injection_onsets=onsets,
        precedence=rules["event_precedence"],
        config=config,
    )
    records = [{"run_id": annotation["run_id"], **record} for record in records]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise SystemExit("No decision times exist after the required history interval")
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} decisions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

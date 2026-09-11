#!/usr/bin/env python3
"""Freeze an alarm threshold from validation predictions under the clean-mission budget."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import select_validation_threshold


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alarm", type=Path, default=ROOT / "configs/alarm_policy.yaml")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.predictions.open(newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit("validation prediction table is empty")
    if {row.get("split") for row in rows} != {"validation"}:
        raise SystemExit("threshold selection accepts validation rows only")
    if any(truth(row.get("protected_test_used", "false")) for row in rows):
        raise SystemExit("protected outcomes cannot be used for threshold selection")
    episodes: dict[str, list[dict[str, str]]] = {}
    clean = set()
    for row in rows:
        run_id = row["run_id"]
        episodes.setdefault(run_id, []).append(row)
        if row.get("fault_family") == "none":
            clean.add(run_id)
    for episode_rows in episodes.values():
        episode_rows.sort(key=lambda row: float(row["decision_time"]))
    alarm = yaml.safe_load(args.alarm.read_text(encoding="utf-8"))
    persistence = alarm["persistence"]
    report = select_validation_threshold(
        episodes,
        clean,
        false_alert_budget=float(alarm["false_alert_budget_per_clean_mission"]),
        required_above=int(persistence["required_above_threshold"]),
        decisions_considered=int(persistence["decisions_considered"]),
        cooldown_seconds=float(alarm["cooldown_seconds"]),
    )
    report.update({
        "schema_version": 1,
        "protocol_version": alarm["protocol_version"],
        "prediction_table": str(args.predictions),
        "protected_test_used": False,
    })
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(f"selected validation threshold {report['threshold']} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

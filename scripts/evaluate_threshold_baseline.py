#!/usr/bin/env python3
"""Apply frozen transparent rules and write immutable event-level results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import AlarmPolicy, apply_alarm_policy, evaluate_event_warnings
from src.models import Rule, ThresholdRuleSet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--rules", type=Path, default=ROOT / "configs/baseline_rules.yaml")
    parser.add_argument("--alarm", type=Path, default=ROOT / "configs/alarm_policy.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.rules.read_text(encoding="utf-8"))
    rules = []
    for name, values in config["rules"].items():
        if values.get("threshold") is None:
            raise SystemExit(f"rule threshold is not frozen: {name}")
        rules.append(Rule(name=name, **values))
    alarm = yaml.safe_load(args.alarm.read_text(encoding="utf-8"))
    if alarm.get("threshold") is None:
        raise SystemExit("alarm threshold is not frozen")
    policy = AlarmPolicy(
        threshold=float(alarm["threshold"]),
        required_above=int(alarm["persistence"]["required_above_threshold"]),
        decisions_considered=int(alarm["persistence"]["decisions_considered"]),
        cooldown_seconds=float(alarm["cooldown_seconds"]),
    )
    features = list(csv.DictReader(args.features.open(newline="", encoding="utf-8")))
    labels = list(csv.DictReader(args.labels.open(newline="", encoding="utf-8")))
    label_by_key = {(row["run_id"], row["decision_index"]): row for row in labels}
    rule_set = ThresholdRuleSet(rules)
    grouped: dict[str, list[dict]] = {}
    for row in features:
        key = (row["run_id"], row["decision_index"])
        if key not in label_by_key:
            raise SystemExit(f"feature row has no label: {key}")
        merged = {**row, **label_by_key[key], **rule_set.predict(row)}
        merged["fired_rules"] = ";".join(merged["fired_rules"])
        grouped.setdefault(row["run_id"], []).append(merged)
    predictions = []
    for run_id, rows in grouped.items():
        rows.sort(key=lambda row: float(row["decision_time"]))
        predictions.extend(apply_alarm_policy(rows, policy))
    if args.predictions.exists() or args.metrics.exists():
        raise SystemExit("refusing to overwrite predictions or metrics")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0].keys()))
        writer.writeheader(); writer.writerows(predictions)
    evaluated: dict[str, list[dict]] = {}
    for row in predictions:
        evaluated.setdefault(row["run_id"], []).append(row)
    args.metrics.write_text(
        json.dumps(evaluate_event_warnings(evaluated), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Recommend, but never execute, guarded recoveries from replay state rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recovery import GuardConfig, RobotState, eligible_actions, rule_matched_action


def boolean(value: str) -> bool:
    if value.lower() not in {"true", "false"}:
        raise ValueError(f"expected true or false, got {value!r}")
    return value.lower() == "true"


def optional_float(value: str) -> float | None:
    return None if value in {"", "none", "null"} else float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config_doc = yaml.safe_load((ROOT / "configs/recovery_guards.yaml").read_text())
    config = GuardConfig(
        minimum_rear_clearance_m=float(config_doc["minimum_rear_clearance_m"]),
        minimum_rotation_clearance_m=float(config_doc["minimum_rotation_clearance_m"]),
        maximum_repeated_recoveries=int(config_doc["maximum_repeated_recoveries"]),
    )
    output = []
    for row in csv.DictReader(args.states.open(newline="", encoding="utf-8")):
        state = RobotState(
            stopped=boolean(row["stopped"]), stop_allowed=boolean(row["stop_allowed"]),
            localisation_poor=boolean(row["localisation_poor"]),
            planning_stale_or_blocked=boolean(row["planning_stale_or_blocked"]),
            rear_clearance_m=optional_float(row["rear_clearance_m"]),
            rotation_clearance_m=optional_float(row["rotation_clearance_m"]),
            immediate_collision_risk=boolean(row["immediate_collision_risk"]),
            obstruction_may_be_transient=boolean(row["obstruction_may_be_transient"]),
            relocalisation_available=boolean(row.get("relocalisation_available", "false")),
            repeated_recovery_count=int(row["repeated_recovery_count"]),
        )
        guards = eligible_actions(state, config)
        action, reason = rule_matched_action(row["diagnosed_signal_group"], guards)
        output.append({
            **row, "recommended_action": action, "selection_reason": reason,
            "eligible_actions": ";".join(name for name, result in guards.items() if result[0]),
            "execution_performed": False,
        })
    if not output:
        raise SystemExit("state table is empty")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0].keys()))
        writer.writeheader(); writer.writerows(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

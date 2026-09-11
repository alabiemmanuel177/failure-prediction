#!/usr/bin/env python3
"""Add alarm/persistent columns to a prediction table under the frozen alarm policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import AlarmPolicy, apply_alarm_policy
from src.evaluation.prediction_tables import (
    confirmatory_gate_passed, group_episodes, guard_protected_rows, policy_settings,
    read_prediction_table, splits_of, write_prediction_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alarm", type=Path, default=ROOT / "configs/alarm_policy.yaml")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="explicit threshold (validation exploration only); otherwise the frozen value",
    )
    parser.add_argument(
        "--secondary-model", default=None,
        help="use this secondary predictor's frozen validation threshold from "
             "configs/model_freeze_secondary.yaml instead of the primary alarm threshold",
    )
    parser.add_argument("--allow-protected-after-freeze", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    secondary_threshold = None
    if args.secondary_model:
        if args.threshold is not None:
            raise SystemExit("--secondary-model and --threshold are mutually exclusive")
        secondary_path = ROOT / "configs/model_freeze_secondary.yaml"
        secondary = yaml.safe_load(secondary_path.read_text(encoding="utf-8")) if secondary_path.exists() else {}
        entry = (secondary or {}).get("secondary_predictors", {}).get(args.secondary_model)
        if not (secondary and secondary.get("frozen") is True and entry):
            raise SystemExit(f"{args.secondary_model} is not a frozen secondary predictor")
        secondary_threshold = float(entry["threshold"])
    rows = read_prediction_table(args.predictions)
    if "alarm" in rows[0]:
        raise SystemExit("input table already carries alarm columns")
    splits = splits_of(rows)
    if splits != {"validation"}:
        if not args.allow_protected_after_freeze:
            raise SystemExit(
                f"prediction table splits {sorted(splits)} are not validation; pass "
                "--allow-protected-after-freeze only after the confirmatory freeze"
            )
        if not confirmatory_gate_passed(ROOT):
            raise SystemExit("check_readiness --stage confirmatory does not pass; refusing")
        if args.threshold is not None:
            raise SystemExit("explicit --threshold is forbidden outside validation tables")
    try:
        protected = guard_protected_rows(
            rows, explicitly_allowed=args.allow_protected_after_freeze, root=ROOT
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    alarm = yaml.safe_load(args.alarm.read_text(encoding="utf-8"))
    settings = policy_settings(alarm)
    threshold = args.threshold if args.threshold is not None else alarm.get("threshold")
    if secondary_threshold is not None:
        threshold = secondary_threshold
    if threshold is None:
        raise SystemExit("alarm threshold is not frozen; pass --threshold for validation exploration")
    policy = AlarmPolicy(
        float(threshold), settings["required_above"], settings["decisions_considered"],
        settings["cooldown_seconds"],
    )
    output = []
    for _run_id, episode_rows in sorted(group_episodes(rows).items()):
        output.extend(apply_alarm_policy(episode_rows, policy))
    digest = write_prediction_table(args.output, output)
    alarms = sum(1 for row in output if row["alarm"])
    print(
        f"threshold={policy.threshold} persistence={policy.required_above}-of-"
        f"{policy.decisions_considered} cooldown={policy.cooldown_seconds}s alarms={alarms} "
        f"protected={'true' if protected else 'false'} -> {args.output} (sha256 {digest[:12]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

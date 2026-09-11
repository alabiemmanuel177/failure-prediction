#!/usr/bin/env python3
"""Fit the R3 cost-sensitive recovery selector from a replay/simulation cost table.

Input CSV columns (one row per warning x candidate action):
  split, map_id, route_id, seed, fault_family, severity, warning_id, risk_score,
  diagnosed_signal_group, stopped, stop_allowed, localisation_poor,
  planning_stale_or_blocked, rear_clearance_m, rotation_clearance_m,
  immediate_collision_risk, obstruction_may_be_transient, relocalisation_available,
  repeated_recovery_count, candidate_action, guard_eligible, observed_cost
Only rows of the fitting split whose action the frozen guard admits are fitted. The
fitting split is ``development`` unless ``--fit-split validation --amendment-id
PA-...`` names an approved protocol amendment that admits validation-map rows (the
post-freeze recovery pilot of PA-2026-09-03-04). Held-out rows are always refused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import sha256_file  # noqa: E402
from src.recovery import GuardConfig  # noqa: E402
from src.recovery.costs import load_cost_config  # noqa: E402
from src.recovery.selector_training import (  # noqa: E402
    evaluate_selector_regret, read_cost_table, save_selector_model, train_selector,
)


def guard_config_from(path: Path) -> GuardConfig:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return GuardConfig(
        minimum_rear_clearance_m=float(document["minimum_rear_clearance_m"]),
        minimum_rotation_clearance_m=float(document["minimum_rotation_clearance_m"]),
        maximum_repeated_recoveries=int(document["maximum_repeated_recoveries"]),
    )


def load_amendment(amendment_id: str, configs: Path = ROOT / "configs") -> dict:
    """Locate an approved amendment that admits selector training on validation rows."""
    for path in sorted(configs.glob("protocol_amendment_*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if document.get("amendment_id") != amendment_id:
            continue
        if document.get("status") != "approved":
            raise ValueError(f"protocol amendment {amendment_id} is not approved")
        source = str(document.get("decisions", {}).get("recovery_selector_training_source", ""))
        if "validation" not in source:
            raise ValueError(
                f"protocol amendment {amendment_id} does not admit validation-map selector training"
            )
        return {
            "amendment_id": amendment_id, "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path), "recovery_selector_training_source": source,
        }
    raise ValueError(f"protocol amendment {amendment_id} is not on file under configs/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cost_table", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="model JSON path (new)")
    parser.add_argument("--guard-config", type=Path, default=ROOT / "configs/recovery_guards.yaml")
    parser.add_argument("--cost-config", type=Path, default=ROOT / "configs/recovery_costs.yaml")
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--minimum-rows-per-action", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--fit-split", choices=("development", "validation"), default="development",
        help="rows of this split are fitted; validation requires --amendment-id",
    )
    parser.add_argument(
        "--amendment-id", default=None,
        help="approved protocol amendment (configs/protocol_amendment_<v>.yaml) admitting the fit split",
    )
    parser.add_argument(
        "--evaluation-table", type=Path,
        help="optional validation-split cost table for selection-only regret reporting",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    guard_config = guard_config_from(args.guard_config)
    cost_document = load_cost_config(args.cost_config)
    rows = read_cost_table(args.cost_table)
    amendment = None
    if args.fit_split != "development":
        if not args.amendment_id:
            raise SystemExit("--fit-split validation requires --amendment-id")
        try:
            amendment = load_amendment(args.amendment_id)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    model = train_selector(
        rows, guard_config, ridge_lambda=args.ridge_lambda,
        minimum_rows_per_action=args.minimum_rows_per_action, seed=args.seed,
        fit_split=args.fit_split, fit_split_admitted_by=args.amendment_id,
    )
    model["inputs"] = {
        "cost_table": str(args.cost_table),
        "cost_table_sha256": sha256_file(args.cost_table),
        "guard_config": str(args.guard_config),
        "guard_config_sha256": sha256_file(args.guard_config),
        "cost_config": str(args.cost_config),
        "cost_config_sha256": sha256_file(args.cost_config),
        "cost_config_status": cost_document.get("status"),
        "fit_split": args.fit_split,
        "amendment": amendment,
    }
    if args.evaluation_table:
        evaluation_rows = read_cost_table(args.evaluation_table)
        splits = {str(row["split"]) for row in evaluation_rows}
        if splits != {"validation"}:
            raise SystemExit(f"evaluation table must contain validation rows only: {sorted(splits)}")
        model["validation_regret"] = {
            "table": str(args.evaluation_table),
            "table_sha256": sha256_file(args.evaluation_table),
            "in_sample": args.fit_split == "validation",
            **evaluate_selector_regret(model, evaluation_rows, guard_config),
        }
    digest = save_selector_model(model, args.output)
    print(json.dumps({
        "model": str(args.output), "sha256": digest,
        "fitted_actions": sorted(model["actions"]),
        "unfitted_actions": model["unfitted_actions"],
        "row_counts": model["row_counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

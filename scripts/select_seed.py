#!/usr/bin/env python3
"""Select, per learned model, the training seed with the highest validation AUPRC.

Reads the validation comparison of each seed pass, applies the preregistered
selection metric (validation AUPRC on eligible decisions), and writes an immutable
seed-selection record reporting every seed's recall at the budget, AUPRC, AUROC,
Brier and ECE so that the spread is visible. It selects nothing on protected data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", action="append", required=True, help="tag=seed, e.g. final_v1=20260903")
    parser.add_argument("--models", default="p3_causal_tcn,p4_gru,p5_compact_transformer")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/model_selection/final_v1/seed_selection.yaml")
    parser.add_argument("--rule", choices=("validation_auprc", "recall_at_budget"), default="validation_auprc",
                        help="recall_at_budget: primary endpoint with AUPRC tie-break (PA-2026-09-04-03)")
    args = parser.parse_args()
    passes = {}
    for item in args.tag:
        tag, seed = item.split("=")
        path = ROOT / "reports/model_selection" / tag / "validation_comparison.yaml"
        passes[tag] = {"seed": int(seed), "path": path, "sha256": sha256_file(path),
                       "document": yaml.safe_load(path.read_text(encoding="utf-8"))}
    selection = {}
    for model_id in args.models.split(","):
        rows = []
        for tag, info in passes.items():
            m = info["document"]["models"].get(model_id)
            if not m:
                continue
            rows.append({
                "tag": tag, "seed": info["seed"],
                "validation_auprc": m["discrimination"]["auprc"],
                "validation_auroc": m["discrimination"]["auroc"],
                "event_recall_at_budget": m["event_recall"],
                "false_alerts_per_clean_mission": m["false_alerts_per_clean_mission"],
                "median_lead_seconds": m["lead_time"]["median_seconds_detected"],
                "brier": m["calibration"]["brier_score"], "ece": m["calibration"]["ece"],
                "threshold": m["threshold_selection"]["threshold"],
            })
        if args.rule == "recall_at_budget":
            best = max(rows, key=lambda r: (r["event_recall_at_budget"], r["validation_auprc"], r["seed"]))
        else:
            best = max(rows, key=lambda r: (r["validation_auprc"], r["seed"]))
        selection[model_id] = {"selected_tag": best["tag"], "selected_seed": best["seed"],
                               "selection_metric": args.rule, "candidates": rows}
    record = {
        "schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation", "protected_test_used": False, "frozen": False,
        "passes": {tag: {"seed": v["seed"], "report": str(v["path"].relative_to(ROOT)), "sha256": v["sha256"]}
                   for tag, v in passes.items()},
        "selection": selection,
        "selection_rule": args.rule,
        "amendment": "PA-2026-09-04-03" if args.rule == "recall_at_budget" else None,
        "note": "Seed selection on validation only; every seed is reported.",
    }
    publish_new_bytes(args.output, yaml.safe_dump(record, sort_keys=False).encode("utf-8"))
    for model_id, item in selection.items():
        print(model_id, "->", item["selected_tag"], "seed", item["selected_seed"],
              [(r["seed"], round(r["validation_auprc"], 3), round(r["event_recall_at_budget"], 3)) for r in item["candidates"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score a derived dataset with the frozen P1 threshold rules into a prediction table.

Uses the thresholds frozen in configs/baseline_rules.yaml (validation-tuned before the
freeze) on the last time step of every all-decisions artifact, and writes the
contract prediction table (raw_score == risk_score in {0, 1}). ``--rules`` points at
another frozen rules file in the same schema, e.g. a leave-one-family-out fold's
``rules_p1.yaml`` written by ``tune_threshold_rules.py --rules-output``. Protected
datasets require --allow-protected-after-freeze and a passing confirmatory gate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import Rule, ThresholdRuleSet  # noqa: E402
from src.protected_data import enforce_protected_boundary  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from tune_threshold_rules import PREDICTION_COLUMNS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rules", type=Path, default=ROOT / "configs/baseline_rules.yaml")
    parser.add_argument("--derived-root", type=Path, default=ROOT / "data/derived")
    parser.add_argument("--allow-protected-after-freeze", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    config = yaml.safe_load(args.rules.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_after_validation_tuning":
        raise SystemExit(f"{args.rules}: rules are not frozen (status must be frozen_after_validation_tuning); "
                         "run tune_threshold_rules.py --write-config (or --rules-output for a fold) first")
    rules = ThresholdRuleSet([
        Rule(name=name, **{k: v for k, v in spec.items() if k in
                           ("feature", "direction", "threshold", "gate_feature", "gate_direction", "gate_threshold")})
        for name, spec in config["rules"].items()
    ])
    root = args.derived_root / args.dataset
    rows = [json.loads(line) for line in (root / "extraction_manifest.jsonl").read_text().splitlines() if line.strip()]
    gate_passed = False
    if any(row["protected_test_used"] is True for row in rows) and args.allow_protected_after_freeze:
        gate_passed = subprocess.run([sys.executable, str(ROOT / "scripts/check_readiness.py"), "--stage", "confirmatory"],
                                     check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    # Fail closed before any output byte exists: a refused run leaves no partial table.
    for row in rows:
        enforce_protected_boundary(row["protected_test_used"], explicitly_allowed=args.allow_protected_after_freeze,
                                   confirmatory_gate_passed=gate_passed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["run_id"]):
            with np.load(root / "decisions" / f"{row['run_id']}.npz", allow_pickle=False) as artifact:
                columns = [str(v) for v in artifact["feature_names"].tolist()]
                last = artifact["X"][:, -1, :]
                eligibility = artifact["eligibility"].tolist()
                index = artifact["decision_index"].tolist()
                times = artifact["decision_time"].tolist()
                y = artifact["y"].tolist()
            for k in range(len(y)):
                record = {column: float(last[k, i]) for i, column in enumerate(columns)}
                score = float(rules.predict(record)["risk_score"])
                writer.writerow({
                    "run_id": row["run_id"], "decision_index": int(index[k]), "decision_time": float(times[k]),
                    "split": row["split"], "map_id": row["map_id"], "route_id": row["route_id"],
                    "fault_family": row["fault_family"], "severity": row["severity"], "seed": row["seed"],
                    "protected_test_used": row["protected_test_used"], "eligibility": str(eligibility[k]),
                    "label": int(y[k]), "primary_event_class": row["primary_event_class"] or "",
                    "primary_event_time": "" if row["primary_event_time"] is None else row["primary_event_time"],
                    "model_id": "p1_threshold_rules", "raw_score": score, "risk_score": score,
                })
                written += 1
    print(f"wrote {written} P1 decisions for {args.dataset} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

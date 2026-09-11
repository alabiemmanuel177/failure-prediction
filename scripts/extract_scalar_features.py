#!/usr/bin/env python3
"""Causally resample long-form scalar telemetry onto an existing decision table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import (
    LeakagePolicy, ScalarSample, extract_decision_rows, load_raw_feature_contract,
    resolve_feature_specs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", type=Path, help="CSV: run_id,timestamp,feature,source,value,max_age_seconds")
    parser.add_argument("decisions", type=Path, help="causal label CSV containing run_id and decision_time")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    decision_rows = list(csv.DictReader(args.decisions.open(newline="", encoding="utf-8")))
    if not decision_rows:
        raise SystemExit("decision table is empty")
    run_ids = {row["run_id"] for row in decision_rows}
    if len(run_ids) != 1:
        raise SystemExit("one extraction invocation must contain exactly one run_id")
    run_id = next(iter(run_ids))
    telemetry_rows = list(csv.DictReader(args.telemetry.open(newline="", encoding="utf-8")))
    if any(row["run_id"] != run_id for row in telemetry_rows):
        raise SystemExit("telemetry contains a different run_id")
    contract = load_raw_feature_contract(ROOT / "configs" / "feature_schema.yaml")
    try:
        specs, grouped = resolve_feature_specs(contract, telemetry_rows)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    samples = {
        name: [ScalarSample(float(row["timestamp"]), float(row["value"])) for row in rows]
        for name, rows in grouped.items()
    }
    policy = LeakagePolicy.from_yaml(ROOT / "configs" / "leakage_denylist.yaml")
    extracted = extract_decision_rows(
        run_id=run_id,
        decision_times=[float(row["decision_time"]) for row in decision_rows],
        specs=specs, samples_by_feature=samples, leakage_policy=policy,
    )
    for source_decision, row in zip(decision_rows, extracted):
        row["decision_index"] = int(source_decision["decision_index"])
        for key in [key for key in row if key.startswith("__audit_source_time__")]:
            source_time = row.pop(key)
            if source_time is not None and float(source_time) > float(row["decision_time"]):
                raise AssertionError(f"causality violation in {key}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(extracted[0].keys()))
        writer.writeheader()
        writer.writerows(extracted)
    print(f"wrote {len(extracted)} causal feature rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

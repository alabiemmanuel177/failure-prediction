#!/usr/bin/env python3
"""Select a calibration method on validation predictions (H3) and publish the artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes, sha256_file
from src.evaluation.calibrators import select_calibration
from src.evaluation.prediction_tables import (
    model_id_of, read_prediction_table, require_validation_only, write_yaml_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="uncalibrated validation prediction table")
    parser.add_argument("--model-id", default=None, help="assert the table's model_id")
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/calibration_policy.yaml")
    parser.add_argument("--alarm", type=Path, default=ROOT / "configs/alarm_policy.yaml")
    parser.add_argument("--output", type=Path, default=None,
                        help="report path; default reports/calibration/<model_id>.yaml")
    parser.add_argument("--calibrator-output", type=Path, default=None,
                        help="calibrator JSON; default reports/calibration/<model_id>.calibrator.json")
    args = parser.parse_args()

    rows = read_prediction_table(args.predictions)
    try:
        require_validation_only(rows, "calibration selection")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    model_id = model_id_of(rows)
    if args.model_id and args.model_id != model_id:
        raise SystemExit(f"table model_id {model_id} does not match --model-id {args.model_id}")
    if any(float(row["risk_score"]) != float(row["raw_score"]) for row in rows):
        raise SystemExit("calibration selection requires an uncalibrated table (risk_score == raw_score)")
    output = args.output or ROOT / "reports/calibration" / f"{model_id}.yaml"
    calibrator_path = args.calibrator_output or ROOT / "reports/calibration" / f"{model_id}.calibrator.json"
    for path in (output, calibrator_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")

    policy = yaml.safe_load(args.policy.read_text(encoding="utf-8"))
    alarm = yaml.safe_load(args.alarm.read_text(encoding="utf-8"))
    selection = select_calibration(rows, policy=policy, alarm=alarm)
    calibrator = selection.pop("calibrator")
    publish_new_bytes(calibrator_path, calibrator.to_json_bytes())
    report = {
        "schema_version": 1,
        "protocol_version": policy.get("protocol_version"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "calibration_id": f"{model_id}:{selection['chosen_method']}:{calibrator.sha256()[:12]}",
        "scope": "validation_only_selection",
        "confirmatory": False,
        "protected_test_used": False,
        "inputs": {
            "prediction_table": str(args.predictions),
            "prediction_table_sha256": sha256_file(args.predictions),
            "calibration_policy": str(args.policy),
            "calibration_policy_sha256": sha256_file(args.policy),
            "alarm_policy": str(args.alarm),
            "alarm_policy_sha256": sha256_file(args.alarm),
        },
        "calibrator_artifact": str(calibrator_path),
        "calibrator_artifact_sha256": sha256_file(calibrator_path),
        **selection,
    }
    write_yaml_report(output, report)
    before, after = report["before"], report["after"]
    print(
        f"{model_id}: chosen {report['chosen_method']} | brier {before['brier_score']:.4f} -> "
        f"{after['brier_score']:.4f} | ece {before['ece']:.4f} -> {after['ece']:.4f} | "
        f"recall@budget {before['event_recall_at_budget']} -> {after['event_recall_at_budget']}"
    )
    print(f"report -> {output}\ncalibrator -> {calibrator_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

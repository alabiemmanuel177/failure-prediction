#!/usr/bin/env python3
"""Assemble the researcher-signed freeze from the seed-selection record.

Steps (idempotent; refuses to overwrite immutable outputs):
  1. freeze the P1 thresholds tuned on the final pool into configs/baseline_rules.yaml
  2. write configs/model_freeze_secondary.yaml naming the selected P4 and P5 seeds
     with their calibrators and thresholds (secondary held-out comparisons)
  3. run scripts/freeze_model.py for the selected primary P3 seed (dry-run unless
     --sign is given), which writes configs/model_freeze.yaml, sets the alarm
     threshold and appends the protocol_change record in the researcher's name
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-selection", type=Path, default=ROOT / "reports/model_selection/final_v1/seed_selection.yaml")
    parser.add_argument("--frozen-by", default="Emmanuel Alabi Olasubomi")
    parser.add_argument("--sign", action="store_true", help="perform the freeze (otherwise dry-run)")
    args = parser.parse_args()
    selection = yaml.safe_load(args.seed_selection.read_text(encoding="utf-8"))["selection"]

    # 1. P1 thresholds from the final-pool tuning record
    rules_path = ROOT / "configs/baseline_rules.yaml"
    config = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    p1_record_path = ROOT / "reports/model_selection/final_v1/p1_threshold_rules.yaml"
    p1_record = yaml.safe_load(p1_record_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_after_validation_tuning":
        if args.sign:
            for name in config["rules"]:
                config["rules"][name]["threshold"] = p1_record["selected_thresholds"][name]
            config["status"] = "frozen_after_validation_tuning"
            config["tuning_record"] = str(p1_record_path.relative_to(ROOT))
            config["tuning_record_sha256"] = sha256_file(p1_record_path)
            config["frozen_utc"] = datetime.now(timezone.utc).isoformat()
            rules_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print("P1 thresholds frozen:", p1_record["selected_thresholds"])
        else:
            print("[dry-run] would freeze P1 thresholds:", p1_record["selected_thresholds"])

    # 2. secondary predictors
    secondary_path = ROOT / "configs/model_freeze_secondary.yaml"
    secondary = {}
    for model_id in ("p4_gru", "p5_compact_transformer"):
        item = selection[model_id]
        tag, seed = item["selected_tag"], item["selected_seed"]
        model_dir = ROOT / "models" / tag / f"{model_id}__seed{seed}"
        calibrator = ROOT / "reports/calibration" / tag / f"{model_id}.calibrator.json"
        threshold = ROOT / "reports/thresholds" / tag / f"{model_id}.yaml"
        secondary[model_id] = {
            "model_dir": str(model_dir.relative_to(ROOT)), "seed": seed, "selected_tag": tag,
            "checkpoint_sha256": (model_dir / "checkpoint.sha256").read_text().split()[0],
            "calibrator": str(calibrator.relative_to(ROOT)), "calibrator_sha256": sha256_file(calibrator),
            "threshold_record": str(threshold.relative_to(ROOT)), "threshold_sha256": sha256_file(threshold),
            "threshold": yaml.safe_load(threshold.read_text(encoding="utf-8"))["threshold"],
            "role": "secondary_held_out_comparison_not_primary",
        }
    if not secondary_path.exists():
        payload = {
            "schema_version": 1, "protocol_version": "1.6", "frozen": args.sign,
            "frozen_by": args.frozen_by, "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "protected_outcomes_consulted": False,
            "seed_selection": str(args.seed_selection.resolve().relative_to(ROOT)),
            "seed_selection_sha256": sha256_file(args.seed_selection.resolve()),
            "secondary_predictors": secondary,
        }
        if args.sign:
            publish_new_bytes(secondary_path, yaml.safe_dump(payload, sort_keys=False).encode("utf-8"))
            print("secondary freeze written:", secondary_path)
        else:
            print("[dry-run] secondary freeze:", {k: (v["selected_tag"], v["seed"], round(v["threshold"], 4)) for k, v in secondary.items()})

    # 3. primary freeze
    item = selection["p3_causal_tcn"]
    tag, seed = item["selected_tag"], item["selected_seed"]
    model_dir = ROOT / "models" / tag / f"p3_causal_tcn__seed{seed}"
    command = [sys.executable, str(ROOT / "scripts/freeze_model.py"),
               "--model-dir", str(model_dir),
               "--calibration-report", str(ROOT / "reports/calibration" / tag / "p3_causal_tcn.yaml"),
               "--threshold-record", str(ROOT / "reports/thresholds" / tag / "p3_causal_tcn.yaml"),
               "--validation-predictions", str(ROOT / "reports/predictions" / tag / "p3_causal_tcn.validation.calibrated.csv"),
               "--frozen-by", args.frozen_by]
    for dataset in ("balanced_pilot_v1", "targeted_development_v1", "development_supplement_v1"):
        command += ["--dataset-manifest", str(ROOT / "data/manifests" / f"{dataset}.dataset.yaml")]
    if not args.sign:
        command.append("--dry-run")
    print("$ " + " ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

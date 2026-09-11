#!/usr/bin/env python3
"""Run the complete pre-freeze model-development sequence for one fitting pool.

Sequence (all validation-only selection, nothing protected, nothing frozen):

  P1 rule tuning -> train P2..P6 -> validation predictions + CPU latency
  -> calibration selection per learned model -> calibrated tables
  -> budget threshold per model -> predictor comparison -> alarm-policy validation

Every step is a subprocess of an existing, individually tested script; outputs are
immutable, so the driver is resumable and skips steps whose outputs exist. The
``--tag`` names the pass (for example ``preliminary_v1`` on the 648/324 pool or
``final_v1`` on the complete fitting pool). The model freeze is deliberately NOT part
of this driver; it is a separate, researcher-signed step.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv/bin/python"
LEARNED = ("p2_reconstruction_ae", "p3_causal_tcn", "p4_gru", "p5_compact_transformer")


def run(command: list[str], log: Path) -> None:
    command = ["nice", "-n", "19", *command]
    with log.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(command, check=False, stdout=stream, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise SystemExit(f"step failed ({result.returncode}); see {log}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--train-dataset", action="append", required=True)
    parser.add_argument("--selection-dataset", required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--models", default=",".join(LEARNED) + ",p6_oracle")
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()
    tag = args.tag
    models = [m for m in args.models.split(",") if m]
    model_root = ROOT / "models" / tag
    predictions = ROOT / "reports/predictions" / tag
    latency_root = ROOT / "reports/latency" / tag
    calibration_root = ROOT / "reports/calibration" / tag
    threshold_root = ROOT / "reports/thresholds" / tag
    selection_root = ROOT / "reports/model_selection" / tag
    for path in (model_root, predictions, latency_root, calibration_root, threshold_root, selection_root):
        path.mkdir(parents=True, exist_ok=True)
    log = ROOT / "logs" / "model_development" / f"{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    p1_record = selection_root / "p1_threshold_rules.yaml"
    p1_table = predictions / "p1_threshold_rules.validation.csv"
    if not p1_record.exists():
        run([sys.executable, str(ROOT / "scripts/tune_threshold_rules.py"),
             *[a for d in args.train_dataset for a in ("--development-dataset", d)],
             "--validation-dataset", args.selection_dataset,
             "--output", str(p1_record), "--predictions", str(p1_table)], log)

    tables: dict[str, Path] = {"p1_threshold_rules": p1_table}
    latencies: dict[str, Path] = {}
    for model_id in models:
        run_name = f"{model_id}__seed{args.seed}"
        model_dir = model_root / run_name
        if not (model_dir / "training_record.json").exists():
            command = [str(VENV), str(ROOT / "scripts/train_predictor.py"), "--model-id", model_id,
                       "--seed", str(args.seed), "--output-root", str(model_root), "--run-name", run_name]
            if model_id != "p6_oracle":
                command += ["--config", str(ROOT / "configs/models" / f"{model_id}.yaml"),
                            *[a for d in args.train_dataset for a in ("--train-dataset", d)],
                            "--selection-dataset", args.selection_dataset]
                if args.max_epochs:
                    command += ["--max-epochs", str(args.max_epochs)]
            run(command, log)
        raw = predictions / f"{model_id}.validation.raw.csv"
        if not raw.exists():
            run([str(VENV), str(ROOT / "scripts/predict_decisions.py"), "--model-dir", str(model_dir),
                 "--dataset", args.selection_dataset, "--output", str(raw)], log)
        if model_id != "p6_oracle":
            latency = latency_root / f"{model_id}.json"
            if not latency.exists():
                run([str(VENV), str(ROOT / "scripts/measure_inference_latency.py"), "--model-dir", str(model_dir),
                     "--dataset", args.selection_dataset, "--output", str(latency)], log)
            latencies[model_id] = latency
            report = calibration_root / f"{model_id}.yaml"
            calibrator = calibration_root / f"{model_id}.calibrator.json"
            if not report.exists():
                run([sys.executable, str(ROOT / "scripts/select_calibration.py"), str(raw),
                     "--model-id", model_id, "--output", str(report), "--calibrator-output", str(calibrator)], log)
            calibrated = predictions / f"{model_id}.validation.calibrated.csv"
            if not calibrated.exists():
                run([sys.executable, str(ROOT / "scripts/apply_calibration.py"), str(raw), str(calibrator),
                     str(calibrated)], log)
            tables[model_id] = calibrated
        else:
            tables[model_id] = raw
        threshold = threshold_root / f"{model_id}.yaml"
        if not threshold.exists():
            run([sys.executable, str(ROOT / "scripts/select_alarm_threshold.py"), str(tables[model_id]),
                 str(threshold)], log)

    comparison = selection_root / "validation_comparison.yaml"
    if not comparison.exists():
        command = [sys.executable, str(ROOT / "scripts/compare_predictors.py"), "--output", str(comparison)]
        for model_id, table in tables.items():
            command += ["--table", f"{model_id}={table}"]
        for model_id, latency in latencies.items():
            command += ["--latency", f"{model_id}={latency}"]
        run(command, log)
    policy_report = selection_root / "alarm_policy_validation.yaml"
    if "p3_causal_tcn" in tables and not policy_report.exists():
        run([sys.executable, str(ROOT / "scripts/validate_alarm_policy.py"), str(tables["p3_causal_tcn"]),
             "--output", str(policy_report)], log)

    summary = {
        "tag": tag, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "train_datasets": args.train_dataset, "selection_dataset": args.selection_dataset,
        "seed": args.seed, "models": models, "protected_test_used": False, "frozen": False,
        "tables": {k: str(v.relative_to(ROOT)) for k, v in tables.items()},
        "comparison": str(comparison.relative_to(ROOT)),
    }
    (selection_root / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"MODEL DEVELOPMENT PASS COMPLETE: {tag} -> {comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

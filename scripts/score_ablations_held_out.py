#!/usr/bin/env python3
"""Score the mandatory ablations on the held-out test episodes (H4).

Protocol: "Ablation deltas using identical test episodes and the primary frozen
threshold rule." ``run_ablations.py`` fits every ablation on the development pool and
reports validation deltas (exploratory). This script takes those fitted models, predicts
the protected held-out dataset (after the freeze, behind the confirmatory gate), applies
each ablation's refit validation calibrator (re-derived deterministically from the same
validation table; its sha256 must equal the recorded one) and the primary frozen
threshold rule (policy ablations apply their own rule to the primary P3 held-out
table), and writes one report per ablation to ``reports/ablations/held_out/<name>.yaml``
in the layout ``evaluate_confirmatory.h4_report`` reads (``ablation``, ``metrics``,
``protected_test_used``). Each report carries the paired hierarchical-bootstrap recall
difference against the primary P3 held-out table over identical episodes.
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
sys.path.insert(0, str(ROOT / "scripts"))

from src.dataset_inventory import sha256_file  # noqa: E402
from src.evaluation.calibrators import apply_calibrator_rows, fit_calibrator  # noqa: E402
from src.evaluation.discrimination import paired_event_recall_difference  # noqa: E402
from src.evaluation.prediction_tables import read_prediction_table, touches_protected  # noqa: E402
from run_ablations import evaluate_rows, load_ablations, policy_for  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", type=Path, default=ROOT / "reports/ablations/validation_ablations.yaml")
    parser.add_argument("--ablations", type=Path, default=ROOT / "configs/ablations.yaml")
    parser.add_argument("--alarm", type=Path, default=ROOT / "configs/alarm_policy.yaml")
    parser.add_argument("--freeze", type=Path, default=ROOT / "configs/model_freeze.yaml")
    parser.add_argument("--held-out-dataset", default="held_out_map_v1-held_out_map_test-1008")
    parser.add_argument("--primary-calibrated", type=Path,
                        default=ROOT / "reports/predictions/held_out/held_out_map_v1/p3_causal_tcn.calibrated.csv")
    parser.add_argument("--primary-alarmed", type=Path,
                        default=ROOT / "reports/predictions/held_out/held_out_map_v1/p3_causal_tcn.alarmed.csv")
    parser.add_argument("--predictions-root", type=Path, default=ROOT / "reports/predictions/held_out/ablations")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/ablations/held_out")
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replicates", type=int, default=2000, help="paired-delta bootstrap replicates")
    parser.add_argument("--discrimination-replicates", type=int, default=0,
                        help="bootstrap replicates for the supporting AUPRC/AUROC intervals (0 = point estimates)")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--allow-protected-after-freeze", action="store_true")
    args = parser.parse_args()

    if not args.allow_protected_after_freeze:
        raise SystemExit("held-out scoring requires --allow-protected-after-freeze")
    gate = subprocess.run([sys.executable, str(ROOT / "scripts/check_readiness.py"), "--stage", "confirmatory"],
                          check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if gate.returncode:
        raise SystemExit("confirmatory readiness gate failed; refusing to touch protected data")

    validation = yaml.safe_load(args.validation_report.read_text(encoding="utf-8"))
    alarm = yaml.safe_load(args.alarm.read_text(encoding="utf-8"))
    freeze = yaml.safe_load(args.freeze.read_text(encoding="utf-8"))
    if validation["inputs"]["freeze_sha256"] != sha256_file(args.freeze):
        raise SystemExit("validation ablation report was produced against a different model freeze")
    if validation["inputs"]["alarm_policy_sha256"] != sha256_file(args.alarm):
        raise SystemExit("validation ablation report was produced against a different alarm policy")
    specs = load_ablations(args.ablations)

    primary_calibrated = read_prediction_table(args.primary_calibrated)
    primary_alarmed = read_prediction_table(args.primary_alarmed)
    if not touches_protected(primary_alarmed):
        raise SystemExit("primary table is not a protected held-out table")
    primary_summary, _ = evaluate_rows(primary_calibrated, policy_for(alarm, {}), alarm=alarm,
                                       replicates=args.discrimination_replicates, seed=args.seed)

    args.predictions_root.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {}
    for name, spec in specs.items():
        output = args.output_dir / f"{name}.yaml"
        if output.exists():
            print(f"{name}: report exists, skipping")
            index[name] = yaml.safe_load(output.read_text(encoding="utf-8"))
            continue
        entry = validation["ablations"].get(name)
        if entry is None:
            raise SystemExit(f"{name}: not present in the validation ablation report")
        report: dict[str, object] = {
            "schema_version": 1, "kind": "held_out_ablation", "ablation": name, "spec": spec,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "held_out_map_test", "protected_test_used": True, "confirmatory": False,
            "hypothesis": "H4 (supporting): identical held-out test episodes, primary frozen threshold rule",
            "held_out_dataset": args.held_out_dataset,
        }
        if spec.get("retrain"):
            model_dir = Path(entry["model_dir"])
            raw = args.predictions_root / f"{name}.raw.csv"
            if not raw.exists():
                command = [args.python, str(ROOT / "scripts/predict_decisions.py"), "--model-dir", str(model_dir),
                           "--dataset", args.held_out_dataset, "--output", str(raw), "--device", args.device,
                           "--allow-protected-after-freeze"]
                print("$", " ".join(command), flush=True)
                subprocess.run(command, check=True)
            validation_raw = read_prediction_table(Path(entry["raw_prediction_table"]))
            calibrator = fit_calibrator(entry["calibrator"]["method"], validation_raw)
            if calibrator.sha256() != entry["calibrator"]["sha256"]:
                raise SystemExit(f"{name}: refit calibrator sha256 differs from the validation record")
            rows = apply_calibrator_rows(read_prediction_table(raw), calibrator)
            policy = policy_for(alarm, spec)
            report.update({
                "model_dir": str(model_dir), "checkpoint_sha256": sha256_file(model_dir / "checkpoint.pt"),
                "raw_prediction_table": str(raw), "raw_prediction_table_sha256": sha256_file(raw),
                "calibrator": {"method": calibrator.method, "sha256": calibrator.sha256(),
                               "refit_from": entry["raw_prediction_table"]},
                "calibration_mode": "refit_on_validation",
            })
        elif name == "no_calibration" or spec.get("calibration") in {"none", False}:
            rows = [{**row, "risk_score": float(row["raw_score"])} for row in primary_calibrated]
            policy = policy_for(alarm, spec)
            report.update({"source_table": str(args.primary_calibrated), "calibration_mode": "none"})
        else:
            rows = primary_calibrated
            if name == "no_persistence" and not spec.get("persistence"):
                spec = {**spec, "persistence": {"required_above_threshold": 1, "decisions_considered": 1}}
                report["spec"] = spec
            policy = policy_for(alarm, spec)
            report.update({"source_table": str(args.primary_calibrated), "calibration_mode": "frozen"})
        summary, alarmed = evaluate_rows(rows, policy, alarm=alarm,
                                         replicates=args.discrimination_replicates, seed=args.seed)
        delta = paired_event_recall_difference(alarmed, primary_alarmed, replicates=args.replicates, seed=args.seed)
        report["metrics"] = {
            **summary,
            "confidence_interval": delta.get("confidence_interval"),
            "confidence_interval_meaning": "95% hierarchical bootstrap interval of the paired event-recall "
                                           "difference (ablation minus primary) over identical held-out episodes",
        }
        report["delta_vs_primary"] = delta
        report["primary"] = {
            "table": str(args.primary_alarmed), "table_sha256": sha256_file(args.primary_alarmed),
            "event_recall": primary_summary["event_recall"],
            "false_alerts_per_clean_mission": primary_summary["false_alerts_per_clean_mission"],
        }
        report["inputs"] = {
            "validation_report": str(args.validation_report), "validation_report_sha256": sha256_file(args.validation_report),
            "freeze_sha256": sha256_file(args.freeze), "alarm_policy_sha256": sha256_file(args.alarm),
            "ablations_config_sha256": sha256_file(args.ablations),
        }
        with output.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(report, stream, sort_keys=False)
        index[name] = report
        print(f"{name}: held-out recall={summary['event_recall']:.4f} fa/clean={summary['false_alerts_per_clean_mission']:.4f} "
              f"delta={delta.get('point_estimate')} ci={delta.get('confidence_interval')} -> {output}")

    summary_path = args.output_dir.parent / "held_out_ablations.yaml"
    payload = {
        "schema_version": 1, "kind": "held_out_ablations_index", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protected_test_used": True, "primary": primary_summary["event_recall"],
        "ablations": {name: {"event_recall": r["metrics"]["event_recall"],
                             "false_alerts_per_clean_mission": r["metrics"]["false_alerts_per_clean_mission"],
                             "delta_vs_primary": r["delta_vs_primary"].get("point_estimate"),
                             "confidence_interval": r["delta_vs_primary"].get("confidence_interval"),
                             "report": str(args.output_dir / f"{name}.yaml")} for name, r in index.items()},
    }
    summary_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print("index ->", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

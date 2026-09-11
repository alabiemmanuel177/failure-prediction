#!/usr/bin/env python3
"""Run the mandatory ablations of a frozen primary predictor and evaluate on validation.

Ablations are read from configs/ablations.yaml (mapping ``ablations: {name: spec}``; a
spec with ``kind: policy`` or ``retrain: false`` reuses the frozen model's validation
table, every other spec retrains via ``scripts/train_predictor.py --ablation <name>`` and
scores validation with ``scripts/predict_decisions.py``). Every variant is evaluated at
the frozen threshold rule (tau, persistence and cooldown from configs/alarm_policy.yaml).
Retrained variants are calibrated with the frozen calibration *method* refitted on
validation (``--calibration-mode refit``, default), the frozen calibrator artifact
(``frozen``) or left raw (``none``). ``--dry-run`` prints the subprocess commands.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import sha256_file
from src.evaluation import AlarmPolicy, apply_alarm_policy, evaluate_event_warnings, reliability_curve
from src.evaluation.calibrators import Calibrator, apply_calibrator_rows, fit_calibrator
from src.evaluation.discrimination import (
    alert_burden_summary, discrimination_from_rows, discrimination_with_intervals,
    lead_time_summary, paired_event_recall_difference,
)
from src.evaluation.prediction_tables import (
    group_episodes, policy_settings, read_prediction_table, require_validation_only,
    write_prediction_table, write_yaml_report,
)

POLICY_ABLATIONS = {"no_calibration", "no_persistence"}


def load_ablations(path: Path) -> dict[str, dict]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = document.get("ablations", document) if isinstance(document, dict) else document
    if isinstance(entries, list):
        entries = {
            (item if isinstance(item, str) else item["name"]):
            ({} if isinstance(item, str) else {k: v for k, v in item.items() if k != "name"})
            for item in entries
        }
    if not isinstance(entries, dict) or not entries:
        raise SystemExit(f"{path}: expected a non-empty 'ablations' mapping")
    output = {}
    for name, spec in entries.items():
        spec = dict(spec or {})
        retrain = spec.get("retrain")
        if retrain is None:
            retrain = spec.get("kind") != "policy" and name not in POLICY_ABLATIONS
        spec["retrain"] = bool(retrain)
        output[str(name)] = spec
    return output


def policy_for(alarm: dict, spec: dict) -> AlarmPolicy:
    settings = policy_settings(alarm)
    persistence = spec.get("persistence") or {}
    return AlarmPolicy(
        float(alarm["threshold"]),
        int(persistence.get("required_above_threshold", settings["required_above"])),
        int(persistence.get("decisions_considered", settings["decisions_considered"])),
        float(spec.get("cooldown_seconds", settings["cooldown_seconds"])),
    )


def evaluate_rows(rows, policy: AlarmPolicy, *, alarm: dict, replicates: int, seed: int):
    episodes = group_episodes(rows)
    alarmed = [row for _run, episode in sorted(episodes.items()) for row in apply_alarm_policy(episode, policy)]
    metrics = evaluate_event_warnings(group_episodes(alarmed))
    clean = {run for run, episode in episodes.items() if episode[0].get("fault_family") == "none"}
    clean_false = sum(e["false_alerts"] for e in metrics["per_episode"] if e["run_id"] in clean)
    curve = reliability_curve(rows, bins=10)
    discrimination = (
        discrimination_with_intervals(rows, replicates=replicates, seed=seed)
        if replicates > 0 else discrimination_from_rows(rows)
    )
    summary = {
        "policy": {
            "threshold": policy.threshold,
            "persistence": f"{policy.required_above}-of-{policy.decisions_considered}",
            "cooldown_seconds": policy.cooldown_seconds,
        },
        "episode_count": metrics["episode_count"],
        "event_count": metrics["event_count"],
        "detected_event_count": metrics["detected_event_count"],
        "event_recall": metrics["event_recall"],
        "false_alerts_per_clean_mission": clean_false / len(clean) if clean else None,
        "false_alerts_per_non_event_mission": metrics["false_alerts_per_non_event_mission"],
        "lead_time": lead_time_summary(metrics),
        "discrimination": discrimination,
        "calibration": {"brier_score": curve["brier_score"], "ece": curve["ece"]},
        "alert_burden": alert_burden_summary(alarmed, decision_rate_hz=float(alarm.get("decision_rate_hz", 2.0))),
    }
    return summary, alarmed


def run(command: list[str], dry_run: bool) -> None:
    print("$", " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(f"command failed with exit code {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=ROOT / "configs/model_freeze.yaml")
    parser.add_argument("--alarm", type=Path, default=ROOT / "configs/alarm_policy.yaml")
    parser.add_argument("--ablations", type=Path, default=ROOT / "configs/ablations.yaml")
    parser.add_argument("--primary-predictions", type=Path, required=True,
                        help="frozen calibrated validation table (raw_score column retained)")
    parser.add_argument("--config", type=Path, required=True, help="primary model config yaml")
    parser.add_argument("--train-dataset", action="append", required=True)
    parser.add_argument("--selection-dataset", required=True)
    parser.add_argument("--validation-dataset", action="append", default=None,
                        help="dataset id(s) scored for evaluation; default = selection dataset")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "models/ablations")
    parser.add_argument("--predictions-root", type=Path, default=ROOT / "reports/predictions/ablations")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/ablations/validation_ablations.yaml")
    parser.add_argument("--calibration-mode", choices=("refit", "frozen", "none"), default="refit")
    parser.add_argument("--replicates", type=int, default=0, help="bootstrap replicates for AUPRC/AUROC (0 = none)")
    parser.add_argument("--only", action="append", default=None, help="restrict to named ablations")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if not args.freeze.exists():
        raise SystemExit(f"model freeze missing: {args.freeze}; ablations run only after G5")
    freeze = yaml.safe_load(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("frozen") is not True or freeze.get("protected_outcomes_consulted") is not False:
        raise SystemExit("freeze record must be frozen with protected outcomes unconsulted")
    alarm = yaml.safe_load(args.alarm.read_text(encoding="utf-8"))
    if alarm.get("threshold") is None:
        raise SystemExit("alarm threshold is not frozen")
    frozen_threshold = freeze.get("alarm_policy", {}).get("threshold")
    if frozen_threshold is not None and float(frozen_threshold) != float(alarm["threshold"]):
        raise SystemExit("alarm_policy.yaml threshold differs from the freeze record")
    ablations = load_ablations(args.ablations)
    if args.only:
        missing = set(args.only) - set(ablations)
        if missing:
            raise SystemExit(f"unknown ablations: {sorted(missing)}")
        ablations = {name: ablations[name] for name in args.only}

    primary_rows = read_prediction_table(args.primary_predictions)
    require_validation_only(primary_rows, "ablation evaluation")
    calibrator_path = Path(str(freeze["calibration"]["artifact"]))
    if not calibrator_path.is_absolute():
        calibrator_path = ROOT / calibrator_path
    frozen_calibrator = Calibrator.from_json_bytes(calibrator_path.read_bytes())
    if frozen_calibrator.sha256() != freeze["calibration"]["artifact_sha256"]:
        raise SystemExit("frozen calibrator artifact sha256 mismatch")
    validation_datasets = args.validation_dataset or [args.selection_dataset]

    primary_policy = policy_for(alarm, {})
    primary_summary, primary_alarmed = evaluate_rows(
        primary_rows, primary_policy, alarm=alarm, replicates=args.replicates, seed=args.seed
    )
    results = {}
    for name, spec in ablations.items():
        print(f"== ablation {name} ({'retrain' if spec['retrain'] else 'policy-only'})")
        entry = {"spec": spec, "retrained": spec["retrain"], "commands": []}
        if spec["retrain"]:
            model_root = args.output_root / name
            train = [
                "nice", "-n", "19", args.python, str(ROOT / "scripts/train_predictor.py"),
                "--model-id", str(freeze["predictor"]["model_id"]), "--config", str(args.config),
            ]
            for dataset in args.train_dataset:
                train += ["--train-dataset", dataset]
            run_name = f"{freeze['predictor']['model_id']}__abl-{name}__seed{args.seed}"
            model_dir = model_root / run_name
            train += ["--selection-dataset", args.selection_dataset, "--ablation", name,
                      "--seed", str(args.seed), "--output-root", str(model_root),
                      "--run-name", run_name]
            complete = (model_dir / "checkpoint.pt").exists() and (model_dir / "training_record.json").exists()
            if model_dir.exists() and not complete and not args.dry_run:
                raise SystemExit(f"incomplete ablation model directory (no checkpoint): {model_dir}; move it aside")
            if complete and not args.dry_run:
                # Resumable after an interruption: a completed fit is reused as is.
                entry["reused_existing_model"] = True
            else:
                run(train, args.dry_run)
            entry["commands"].append(" ".join(shlex.quote(p) for p in train))
            raw_table = args.predictions_root / f"{name}.raw.csv"
            predict = [args.python, str(ROOT / "scripts/predict_decisions.py"), "--model-dir", str(model_dir)]
            for dataset in validation_datasets:
                predict += ["--dataset", dataset]
            predict += ["--output", str(raw_table)]
            if raw_table.exists() and not args.dry_run:
                entry["reused_existing_predictions"] = True  # resumable after an interruption
            else:
                run(predict, args.dry_run)
            entry["commands"].append(" ".join(shlex.quote(p) for p in predict))
            entry["model_dir"] = str(model_dir)
            entry["raw_prediction_table"] = str(raw_table)
            entry["calibration_mode"] = args.calibration_mode
            if args.dry_run:
                print(f"   then: calibrate ({args.calibration_mode}: {frozen_calibrator.method}), "
                      f"apply frozen policy, evaluate on validation")
                results[name] = entry
                continue
            rows = read_prediction_table(raw_table)
            require_validation_only(rows, f"ablation {name}")
            if args.calibration_mode == "refit":
                calibrator = fit_calibrator(frozen_calibrator.method, rows)
            elif args.calibration_mode == "frozen":
                calibrator = frozen_calibrator
            else:
                calibrator = Calibrator("identity")
            rows = apply_calibrator_rows(rows, calibrator)
            calibrated_table = args.predictions_root / f"{name}.calibrated.csv"
            if calibrated_table.exists():
                # Derived, deterministic from the raw table and the frozen method; an
                # interrupted run may have left it, so it is rewritten rather than refused.
                calibrated_table.unlink()
            entry["calibrated_prediction_table"] = str(calibrated_table)
            entry["calibrated_prediction_table_sha256"] = write_prediction_table(calibrated_table, rows)
            entry["calibrator"] = {"method": calibrator.method, "sha256": calibrator.sha256()}
            policy = policy_for(alarm, spec)
        elif name == "no_calibration" or spec.get("calibration") in {"none", False}:
            rows = [{**row, "risk_score": float(row["raw_score"])} for row in primary_rows]
            policy = policy_for(alarm, spec)
            entry["calibration_mode"] = "none"
        else:
            rows = primary_rows
            persistence = spec.get("persistence")
            if name == "no_persistence" and not persistence:
                spec = {**spec, "persistence": {"required_above_threshold": 1, "decisions_considered": 1}}
                entry["spec"] = spec
            policy = policy_for(alarm, spec)
            entry["calibration_mode"] = "frozen"
        summary, alarmed = evaluate_rows(rows, policy, alarm=alarm, replicates=args.replicates, seed=args.seed)
        entry["result"] = summary
        try:
            entry["delta_vs_primary"] = paired_event_recall_difference(
                alarmed, primary_alarmed, replicates=2000, seed=args.seed
            )
        except ValueError as error:
            entry["delta_vs_primary"] = {"error": str(error)}
        results[name] = entry
        print(f"   recall={summary['event_recall']} fa/clean={summary['false_alerts_per_clean_mission']} "
              f"auprc={summary['discrimination']['auprc']}")

    if args.dry_run:
        print(f"DRY RUN: would write {args.output}")
        return 0
    report = {
        "schema_version": 1,
        "protocol_version": alarm.get("protocol_version"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "validation_only",
        "confirmatory": False,
        "protected_test_used": False,
        "note": "Ablation deltas on validation episodes at the primary frozen threshold rule; exploratory.",
        "inputs": {
            "freeze": str(args.freeze), "freeze_sha256": sha256_file(args.freeze),
            "alarm_policy_sha256": sha256_file(args.alarm),
            "ablations_config_sha256": sha256_file(args.ablations),
            "primary_predictions": str(args.primary_predictions),
            "primary_predictions_sha256": sha256_file(args.primary_predictions),
            "train_datasets": args.train_dataset, "selection_dataset": args.selection_dataset,
            "validation_datasets": validation_datasets, "seed": args.seed,
            "calibration_mode": args.calibration_mode,
        },
        "primary": {"model_id": freeze["predictor"]["model_id"], "result": primary_summary},
        "ablations": results,
    }
    write_yaml_report(args.output, report)
    print(f"report -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

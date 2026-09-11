#!/usr/bin/env python3
"""Compare predictors on validation prediction tables (selection only, never confirmatory).

Each model gets its own budget threshold from ``select_validation_threshold`` under the
preregistered persistence and cooldown. Event recall, false alerts, lead time with the
full denominator, AUPRC/AUROC with episode-grouped bootstrap intervals, Brier, ECE,
alert burden, latency and by-family/by-map breakdowns are reported, plus the paired
primary-minus-baseline recall difference (the H1 estimator rehearsed on validation).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import sha256_file
from src.evaluation import AlarmPolicy, apply_alarm_policy, reliability_curve, select_validation_threshold
from src.evaluation.discrimination import (
    alert_burden_summary, discrimination_with_intervals, grouped_event_summary,
    lead_time_summary, paired_event_recall_difference,
)
from src.evaluation.prediction_tables import (
    clean_run_ids, group_episodes, model_id_of, policy_settings, read_prediction_table,
    require_validation_only, write_yaml_report,
)


def parse_pair(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected <model_id>=<path>")
    model_id, path = value.split("=", 1)
    return model_id.strip(), Path(path.strip())


def evaluate_model(
    rows, *, alarm, replicates: int, seed: int, bins: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    settings = policy_settings(alarm)
    episodes = group_episodes(rows)
    clean = clean_run_ids(rows)
    selection = select_validation_threshold(
        episodes, clean, false_alert_budget=settings["false_alert_budget"],
        required_above=settings["required_above"],
        decisions_considered=settings["decisions_considered"],
        cooldown_seconds=settings["cooldown_seconds"],
    )
    policy = AlarmPolicy(
        selection["threshold"], settings["required_above"], settings["decisions_considered"],
        settings["cooldown_seconds"],
    )
    alarmed = [row for _run_id, episode in sorted(episodes.items()) for row in apply_alarm_policy(episode, policy)]
    metrics = selection["metrics"]
    curve = reliability_curve(rows, bins=bins)
    report = {
        "threshold_selection": {
            "selection_split": "validation",
            "threshold": selection["threshold"],
            "false_alert_budget_per_clean_mission": settings["false_alert_budget"],
            "persistence": f"{policy.required_above}-of-{policy.decisions_considered}",
            "cooldown_seconds": policy.cooldown_seconds,
            "candidate_threshold_count": selection["candidate_threshold_count"],
            "feasible_threshold_count": selection["feasible_threshold_count"],
        },
        "episode_count": metrics["episode_count"],
        "clean_mission_count": len(clean),
        "event_count": metrics["event_count"],
        "detected_event_count": metrics["detected_event_count"],
        "undetected_event_count": metrics["undetected_event_count"],
        "event_recall": metrics["event_recall"],
        "false_alert_count": metrics["false_alert_count"],
        "false_alerts_per_clean_mission": selection["false_alerts_per_clean_mission"],
        "false_alerts_per_non_event_mission": metrics["false_alerts_per_non_event_mission"],
        "false_alerts_per_mission": metrics["false_alerts_per_mission"],
        "lead_time": lead_time_summary(metrics),
        "discrimination": discrimination_with_intervals(rows, replicates=replicates, seed=seed),
        "calibration": {
            "brier_score": curve["brier_score"], "ece": curve["ece"],
            "eligible_decision_count": curve["eligible_decision_count"],
            "bin_count": curve["bin_count"],
        },
        "alert_burden": alert_burden_summary(
            alarmed, decision_rate_hz=float(alarm.get("decision_rate_hz", 2.0))
        ),
        "by_family": grouped_event_summary(alarmed, "fault_family"),
        "by_map": grouped_event_summary(alarmed, "map_id"),
        "by_severity": grouped_event_summary(alarmed, "severity"),
    }
    return report, alarmed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=parse_pair, action="append", required=True,
                        metavar="MODEL_ID=PATH", help="uncalibrated or calibrated validation table")
    parser.add_argument("--latency", type=parse_pair, action="append", default=[],
                        metavar="MODEL_ID=PATH", help="optional latency JSON per model")
    parser.add_argument("--primary", default="p3_causal_tcn")
    parser.add_argument("--baseline", default="p1_threshold_rules")
    parser.add_argument("--alarm", type=Path, default=ROOT / "configs/alarm_policy.yaml")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--paired-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/model_selection/validation_comparison.yaml")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    tables = dict(args.table)
    if len(tables) != len(args.table):
        raise SystemExit("duplicate model ids among --table arguments")
    latencies = dict(args.latency)
    alarm = yaml.safe_load(args.alarm.read_text(encoding="utf-8"))

    models = {}
    alarmed_tables = {}
    episode_sets = {}
    for model_id, path in tables.items():
        rows = read_prediction_table(path)
        try:
            require_validation_only(rows, f"predictor comparison ({model_id})")
        except ValueError as error:
            raise SystemExit(str(error)) from error
        table_model = model_id_of(rows)
        if table_model != model_id:
            raise SystemExit(f"{path}: table model_id {table_model} != {model_id}")
        report, alarmed = evaluate_model(
            rows, alarm=alarm, replicates=args.replicates, seed=args.seed, bins=args.bins
        )
        report["prediction_table"] = str(path)
        report["prediction_table_sha256"] = sha256_file(path)
        report["calibrated_input"] = any(
            float(row["risk_score"]) != float(row["raw_score"]) for row in rows
        )
        if model_id in latencies:
            latency_path = latencies[model_id]
            report["latency"] = {
                "source": str(latency_path), "source_sha256": sha256_file(latency_path),
                **json.loads(latency_path.read_text(encoding="utf-8")),
            }
        else:
            report["latency"] = None
        models[model_id] = report
        alarmed_tables[model_id] = alarmed
        episode_sets[model_id] = set(group_episodes(rows))

    paired = None
    if args.primary in alarmed_tables and args.baseline in alarmed_tables:
        try:
            paired = paired_event_recall_difference(
                alarmed_tables[args.primary], alarmed_tables[args.baseline],
                replicates=args.paired_replicates, seed=args.seed,
            )
            paired["contrast"] = f"{args.primary}_minus_{args.baseline}_event_recall"
        except ValueError as error:
            paired = {"error": str(error)}
    shared = set.intersection(*episode_sets.values()) if episode_sets else set()
    report = {
        "schema_version": 1,
        "protocol_version": alarm.get("protocol_version"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "validation_only",
        "confirmatory": False,
        "note": (
            "Validation-only model selection. Thresholds are re-selected per model under "
            "the preregistered budget; nothing here is a held-out-map or confirmatory result."
        ),
        "protected_test_used": False,
        "alarm_policy": str(args.alarm),
        "alarm_policy_sha256": sha256_file(args.alarm),
        "bootstrap": {
            "hierarchy": ["map", "route", "episode"], "replicates": args.replicates,
            "paired_replicates": args.paired_replicates, "seed": args.seed,
        },
        "episode_overlap": {
            "shared_episode_count": len(shared),
            "identical_episode_sets": all(episodes == shared for episodes in episode_sets.values()),
        },
        "models": models,
        "paired_primary_minus_baseline": {
            "primary": args.primary, "baseline": args.baseline, "estimator": "H1_rehearsal_validation",
            "result": paired,
        },
    }
    write_yaml_report(args.output, report)
    for model_id, entry in models.items():
        disc = entry["discrimination"]
        print(
            f"{model_id:>24} tau={entry['threshold_selection']['threshold']:.4f} "
            f"recall={entry['event_recall']} ({entry['detected_event_count']}/{entry['event_count']}) "
            f"fa/clean={entry['false_alerts_per_clean_mission']:.3f} "
            f"lead_med={entry['lead_time']['median_seconds_detected']} "
            f"auprc={disc['auprc']} brier={entry['calibration']['brier_score']:.4f} "
            f"ece={entry['calibration']['ece']:.4f}"
        )
    if paired and "point_estimate" in paired:
        print(
            f"paired {args.primary}-{args.baseline} recall diff = {paired['point_estimate']:.4f} "
            f"CI {paired['confidence_interval']} (validation only)"
        )
    print(f"report -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

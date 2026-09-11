#!/usr/bin/env python3
"""Tune and freeze the P1 transparent threshold rules under the alarm budget.

Protocol 1.0 §9 defines P1 as validation-tuned rules on covariance, progress, scan
health and oscillation. Candidate thresholds for each rule are development-data
quantiles of the corresponding feature (so validation never defines the search
space); the four thresholds are then chosen by deterministic coordinate ascent on
validation episodes to maximise event recall subject to at most the frozen number of
false alerts per clean validation mission under the frozen 2-of-3 / 10 s alarm policy.
The result is written as an immutable rule record and, on request, into
`configs/baseline_rules.yaml`. Rules operate on the last time step of the
all-decisions artifacts so P1 sees exactly the deployable feature stream.

Leave-one-family-out folds (``scripts/run_unseen_family_folds.py``) pass
``--exclude-family F``: family-F episodes are removed from both the development
quantile grid and the validation tuning set, the exclusion is recorded in the tuning
record, and ``--rules-output`` writes a per-fold frozen rules file in the
``configs/baseline_rules.yaml`` schema instead of touching ``configs/``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402
from src.evaluation import AlarmPolicy, apply_alarm_policy, evaluate_event_warnings  # noqa: E402
from src.models import Rule, ThresholdRuleSet  # noqa: E402

PREDICTION_COLUMNS = (
    "run_id", "decision_index", "decision_time", "split", "map_id", "route_id",
    "fault_family", "severity", "seed", "protected_test_used", "eligibility", "label",
    "primary_event_class", "primary_event_time", "model_id", "raw_score", "risk_score",
)


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def load_dataset(dataset_id: str, derived_root: Path = ROOT / "data/derived") -> tuple[list[dict], dict[str, list[dict]]]:
    root = derived_root / dataset_id
    rows = [json.loads(line) for line in (root / "extraction_manifest.jsonl").read_text().splitlines() if line.strip()]
    episodes: dict[str, list[dict]] = {}
    for row in rows:
        if row["protected_test_used"] is not False:
            raise SystemExit(f"refusing protected episode {row['run_id']}")
        with np.load(root / "decisions" / f"{row['run_id']}.npz", allow_pickle=False) as artifact:
            columns = [str(value) for value in artifact["feature_names"].tolist()]
            last = artifact["X"][:, -1, :]
            eligibility = artifact["eligibility"].tolist()
            decision_index = artifact["decision_index"].tolist()
            decision_time = artifact["decision_time"].tolist()
            y = artifact["y"].tolist()
        decisions = []
        for k in range(len(y)):
            record = {column: float(last[k, i]) for i, column in enumerate(columns)}
            record.update({
                "run_id": row["run_id"], "decision_index": int(decision_index[k]),
                "decision_time": float(decision_time[k]), "eligibility": str(eligibility[k]),
                "label": int(y[k]), "primary_event_time": row["primary_event_time"],
                "primary_event_class": row["primary_event_class"],
            })
            decisions.append(record)
        episodes[row["run_id"]] = decisions
    return rows, episodes


def candidate_thresholds(episodes: dict[str, list[dict]], feature: str, quantiles: np.ndarray) -> list[float]:
    values = np.asarray([
        decision[feature] for rows in episodes.values() for decision in rows
        if int(decision.get(f"{feature}__missing", 0)) == 0
    ])
    if values.size == 0:
        raise SystemExit(f"no observed development values for {feature}")
    grid = {float(round(v, 6)) for v in np.quantile(values, quantiles)}
    # Never-fire sentinels lie strictly outside the observed development range so
    # coordinate ascent can start from a rule that is switched off.
    span = float(values.max() - values.min()) or 1.0
    grid.add(float(values.max() + span))
    grid.add(float(values.min() - span))
    return sorted(grid)


def evaluate(rule_specs: dict, thresholds: dict[str, float], episodes: dict[str, list[dict]],
             clean_ids: set[str], policy: AlarmPolicy) -> dict:
    rules = [Rule(name=name, **{**spec, "threshold": thresholds[name]}) for name, spec in rule_specs.items()]
    rule_set = ThresholdRuleSet(rules)
    predicted = {}
    for run_id, rows in episodes.items():
        scored = [{**row, **rule_set.predict(row)} for row in rows]
        predicted[run_id] = apply_alarm_policy(scored, policy)
    metrics = evaluate_event_warnings(predicted)
    clean_false = sum(item["false_alerts"] for item in metrics["per_episode"] if item["run_id"] in clean_ids)
    return {
        "event_recall": metrics["event_recall"],
        "false_alerts_per_clean_mission": clean_false / len(clean_ids),
        "false_alerts_per_mission": metrics["false_alerts_per_mission"],
        "median_useful_lead_seconds_detected": metrics["median_useful_lead_seconds_detected"],
        "detected_event_count": metrics["detected_event_count"],
        "event_count": metrics["event_count"],
    }


def ranking(item: dict) -> tuple:
    recall = item["event_recall"]
    lead = item["median_useful_lead_seconds_detected"]
    return (
        float(recall) if recall is not None else -1.0,
        -float(item["false_alerts_per_clean_mission"]),
        float(lead) if lead is not None else -1.0,
    )


def exclude_family(rows: list[dict], episodes: dict[str, list[dict]], family: str, split: str) -> int:
    """Drop every episode of ``family`` in place; refuse a silent no-op fold."""
    dropped = {row["run_id"] for row in rows if row["fault_family"] == family}
    if not dropped:
        raise SystemExit(f"--exclude-family {family!r} matches no {split} episode; refusing silent no-op fold")
    for run_id in dropped:
        episodes.pop(run_id)
    rows[:] = [row for row in rows if row["run_id"] not in dropped]
    return len(dropped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dataset", action="append", required=True)
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/model_selection/p1_threshold_rules.yaml")
    parser.add_argument("--predictions", type=Path, default=ROOT / "reports/predictions/p1_threshold_rules.validation.csv")
    parser.add_argument("--write-config", action="store_true", help="freeze thresholds into configs/baseline_rules.yaml")
    parser.add_argument("--exclude-family", default=None,
                        help="leave-one-family-out fold: drop this family's episodes from the development "
                             "quantile grid and the validation tuning set")
    parser.add_argument("--rules-output", type=Path, default=None,
                        help="write a frozen per-fold rules file (configs/baseline_rules.yaml schema) here "
                             "instead of touching configs/")
    parser.add_argument("--rules", type=Path, default=ROOT / "configs/baseline_rules.yaml",
                        help="rule specification (features, directions, gates); thresholds are ignored")
    parser.add_argument("--alarm", type=Path, default=ROOT / "configs/alarm_policy.yaml")
    parser.add_argument("--derived-root", type=Path, default=ROOT / "data/derived")
    parser.add_argument("--quantile-grid", type=int, default=25)
    parser.add_argument("--sweeps", type=int, default=3)
    args = parser.parse_args(argv)
    if args.output.exists() or args.predictions.exists():
        raise SystemExit("refusing to overwrite P1 tuning outputs")
    if args.rules_output is not None and args.rules_output.exists():
        raise SystemExit(f"refusing to overwrite per-fold rules {args.rules_output}")
    if args.write_config and (args.exclude_family or args.rules_output is not None):
        raise SystemExit("--write-config is reserved for the full-data P1 freeze; per-fold rules go to --rules-output")
    if args.exclude_family == "none":
        raise SystemExit("clean episodes define the false-alert budget and cannot be excluded")
    config = yaml.safe_load(args.rules.read_text(encoding="utf-8"))
    alarm = yaml.safe_load(args.alarm.read_text(encoding="utf-8"))
    policy = AlarmPolicy(
        threshold=0.5, required_above=int(alarm["persistence"]["required_above_threshold"]),
        decisions_considered=int(alarm["persistence"]["decisions_considered"]),
        cooldown_seconds=float(alarm["cooldown_seconds"]),
    )
    budget = float(alarm["false_alert_budget_per_clean_mission"])
    rule_specs = {name: {k: v for k, v in spec.items() if k != "threshold"} for name, spec in config["rules"].items()}

    development_rows: list[dict] = []
    development: dict[str, list[dict]] = {}
    for dataset_id in args.development_dataset:
        rows, episodes = load_dataset(dataset_id, args.derived_root)
        if any(row["split"] != "development" for row in rows):
            raise SystemExit("development datasets must be development split")
        development_rows.extend(rows)
        development.update(episodes)
    validation_rows, validation = load_dataset(args.validation_dataset, args.derived_root)
    if any(row["split"] != "validation" for row in validation_rows):
        raise SystemExit("validation dataset must be validation split")
    excluded_counts = None
    if args.exclude_family:
        excluded_counts = {
            "development": exclude_family(development_rows, development, args.exclude_family, "development"),
            "validation": exclude_family(validation_rows, validation, args.exclude_family, "validation"),
        }
    clean_ids = {row["run_id"] for row in validation_rows if row["fault_family"] == "none"}
    quantiles = np.unique(np.concatenate([
        np.linspace(0.02, 0.98, args.quantile_grid),
        [0.001, 0.005, 0.01, 0.99, 0.995, 0.999],
    ]))
    candidates = {name: candidate_thresholds(development, spec["feature"], quantiles)
                  for name, spec in rule_specs.items()}
    # Start from the most conservative (never-firing) end of each rule.
    thresholds = {
        name: (candidates[name][-1] if spec["direction"] == "above" else candidates[name][0])
        for name, spec in rule_specs.items()
    }
    best = evaluate(rule_specs, thresholds, validation, clean_ids, policy)
    if best["false_alerts_per_clean_mission"] > budget:
        raise SystemExit("even the most conservative rules exceed the clean-mission budget")
    trace = []
    for sweep in range(args.sweeps):
        improved = False
        for name in sorted(rule_specs):
            for candidate in candidates[name]:
                trial = {**thresholds, name: candidate}
                result = evaluate(rule_specs, trial, validation, clean_ids, policy)
                if result["false_alerts_per_clean_mission"] <= budget and ranking(result) > ranking(best):
                    thresholds, best, improved = trial, result, True
                    trace.append({"sweep": sweep, "rule": name, "threshold": candidate, **result})
        if not improved:
            break
    development_metrics = evaluate(rule_specs, thresholds, development, {
        run_id for run_id, rows in development.items() if all(r["primary_event_time"] is None for r in rows)
    } or set(development), policy)

    rules = ThresholdRuleSet([Rule(name=name, **{**spec, "threshold": thresholds[name]}) for name, spec in rule_specs.items()])
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    meta = {row["run_id"]: row for row in validation_rows}
    with args.predictions.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for run_id in sorted(validation):
            row = meta[run_id]
            for decision in validation[run_id]:
                score = float(rules.predict(decision)["risk_score"])
                writer.writerow({
                    "run_id": run_id, "decision_index": decision["decision_index"],
                    "decision_time": decision["decision_time"], "split": row["split"],
                    "map_id": row["map_id"], "route_id": row["route_id"],
                    "fault_family": row["fault_family"], "severity": row["severity"],
                    "seed": row["seed"], "protected_test_used": False,
                    "eligibility": decision["eligibility"], "label": decision["label"],
                    "primary_event_class": row["primary_event_class"] or "",
                    "primary_event_time": "" if row["primary_event_time"] is None else row["primary_event_time"],
                    "model_id": "p1_threshold_rules", "raw_score": score, "risk_score": score,
                })
    record = {
        "schema_version": 1, "model_id": "p1_threshold_rules",
        "selection_split": "validation", "protected_test_used": False,
        "tuned_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "maximum_validation_event_recall_subject_to_false_alert_budget",
        "false_alert_budget_per_clean_mission": budget,
        "alarm_policy": {"required_above": policy.required_above,
                         "decisions_considered": policy.decisions_considered,
                         "cooldown_seconds": policy.cooldown_seconds,
                         "score_threshold": policy.threshold},
        "candidate_source": "development_feature_quantiles",
        "candidate_thresholds": candidates,
        "selected_thresholds": thresholds,
        "validation_metrics": best,
        "development_metrics_informational": development_metrics,
        "search_trace": trace,
        "development_datasets": args.development_dataset,
        "validation_dataset": args.validation_dataset,
        "excluded_family": args.exclude_family,
        "excluded_episode_counts": excluded_counts,
        "development_episode_count": len(development),
        "validation_episode_count": len(validation),
        "validation_predictions": display_path(args.predictions),
        "validation_predictions_sha256": sha256_file(args.predictions),
        "rules_output": None if args.rules_output is None else display_path(args.rules_output),
        "note": "P1 is a transparent baseline; these thresholds are frozen before any protected evaluation.",
    }
    if args.exclude_family:
        record["note"] = (
            f"leave-one-family-out fold: {args.exclude_family} episodes were removed from the "
            "development quantile grid and the validation tuning set; thresholds are frozen before "
            "any protected evaluation of that family."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    publish_new_bytes(args.output, yaml.safe_dump(record, sort_keys=False).encode("utf-8"))
    if args.rules_output is not None:
        fold_config = {
            key: value for key, value in config.items()
            if key not in ("tuning_record", "tuning_record_sha256", "tuned_utc", "frozen_utc")
        }
        fold_config["status"] = "frozen_after_validation_tuning"
        fold_config["rules"] = {
            name: {**config["rules"][name], "threshold": thresholds[name]} for name in config["rules"]
        }
        fold_config["excluded_family"] = args.exclude_family
        fold_config["tuning_record"] = display_path(args.output)
        fold_config["tuning_record_sha256"] = sha256_file(args.output)
        fold_config["frozen_utc"] = record["tuned_utc"]
        publish_new_bytes(args.rules_output, yaml.safe_dump(fold_config, sort_keys=False).encode("utf-8"))
    if args.write_config:
        for name in config["rules"]:
            config["rules"][name]["threshold"] = thresholds[name]
        config["status"] = "frozen_after_validation_tuning"
        config["tuning_record"] = display_path(args.output)
        config["tuned_utc"] = record["tuned_utc"]
        (ROOT / "configs/baseline_rules.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps({"selected_thresholds": thresholds, **best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

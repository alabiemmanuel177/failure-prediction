#!/usr/bin/env python3
"""Evaluate the sequential-versus-parallel health-feature shift check.

Compares, per matched condition (clean S0, clean S3, medium planner oscillation on
S0), the deployable feature distributions on eligible decisions of the six-worker
check dataset against the sequential development dataset: standardised mean
differences and quantile shifts for every deployable column (28 primary values x
{value, age_seconds, missing}), with ``inference_latency_ms``, every
``*__age_seconds`` channel and ``valid_return_fraction`` highlighted, plus
per-episode message-rate proxies from the raw scalar telemetry.

When a frozen-model prediction table and the frozen threshold record are supplied,
the clean-mission false alerts per mission under the frozen alarm policy are
computed on the check episodes and on the sequential development clean episodes,
pooled over both clean conditions (the pass rule) and broken down per condition
(clean S0 and clean S3 separately, so a system-specific shift is visible).

The pre-declared pass rule lives in ``configs/concurrency_shift_policy.yaml`` and is
the same for every check variant. ``--check-dataset-id`` names the derived check
dataset (``<campaign_id>-development-<n>``); the report's ``check_campaign_id`` and
default output path (``reports/integrity/<campaign_id>.yaml``) follow from it, so the
v2 check (``concurrency_shift_check_v2-development-36``) is evaluated with the same
policy and never overwrites the v1 report. The report is immutable and carries
``passed: true|false``.

    python3 scripts/evaluate_concurrency_shift.py \\
        --check-dataset-id concurrency_shift_check_v1-development-36 \\
        [--predictions reports/predictions/<frozen>.csv \\
         --threshold-record reports/model_selection/<final>/threshold.yaml]
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_inventory import sha256_file  # noqa: E402
from src.evaluation.event_metrics import evaluate_event_warnings  # noqa: E402
from src.evaluation.policy import AlarmPolicy, apply_alarm_policy  # noqa: E402
from src.evaluation.prediction_tables import (  # noqa: E402
    group_episodes, guard_protected_rows, policy_settings, read_prediction_table,
    write_yaml_report,
)

ELIGIBLE = {"eligible_positive", "eligible_negative"}
CONDITIONS = {
    ("none", "s0"): "clean_s0",
    ("none", "s3"): "clean_s3",
    ("planner_oscillation", "s0"): "oscillation",
}
CLEAN_CONDITIONS = ("clean_s0", "clean_s3")


def check_campaign_id_of(dataset_id: str | None, policy: dict) -> str | None:
    """Campaign id behind a derived check dataset id (``<campaign>-<split>-<n>``).

    Falls back to the policy's ``check_campaign_id`` when no dataset id is given or it
    does not carry the ``-development-`` segment.
    """
    if isinstance(dataset_id, str) and "-development-" in dataset_id:
        return dataset_id.split("-development-", 1)[0]
    return policy.get("check_campaign_id")


# --------------------------------------------------------------------------- inputs
def load_manifest_rows(dataset_root: Path) -> list[dict]:
    path = dataset_root / "extraction_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"extraction manifest missing: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row.get("protected_test_used") is not False or row.get("split") != "development":
            raise ValueError(
                f"{path}: refusing non-development or protected episode {row.get('run_id')}"
            )
    return rows


def condition_of(row: dict) -> str | None:
    system = str(row.get("system_id") or "").lower()
    return CONDITIONS.get((str(row.get("fault_family")), system))


def deployable_columns(schema: dict) -> list[str]:
    values = list(schema["primary_feature_set"]["values"])
    companions = list(schema["primary_feature_set"].get("include_companion_channels", []))
    columns: list[str] = []
    for name in values:
        columns.append(name)
        for companion in companions:
            columns.append(f"{name}__{companion}")
    return columns


def eligible_decisions(dataset_root: Path, run_id: str) -> set[int]:
    with (dataset_root / "labels" / f"{run_id}.csv").open(newline="", encoding="utf-8") as stream:
        return {
            int(row["decision_index"]) for row in csv.DictReader(stream)
            if row["eligibility"] in ELIGIBLE
        }


def feature_rows(dataset_root: Path, run_id: str, columns: list[str]) -> dict[str, list[float]]:
    eligible = eligible_decisions(dataset_root, run_id)
    values: dict[str, list[float]] = {column: [] for column in columns}
    with (dataset_root / "window_features" / f"{run_id}.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [column for column in columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{run_id}: window features lack deployable columns {missing[:5]}")
        for row in reader:
            if int(row["decision_index"]) not in eligible:
                continue
            for column in columns:
                values[column].append(float(row[column]))
    return values


def label_span(dataset_root: Path, run_id: str) -> tuple[float, float]:
    with (dataset_root / "labels" / f"{run_id}.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{run_id}: empty label grid")
    return (
        min(float(row["window_start"]) for row in rows),
        max(float(row["decision_time"]) for row in rows),
    )


def message_rates(dataset_root: Path, run_id: str, sources: dict[str, str]) -> dict[str, float]:
    """Receive rate (Hz) of one representative scalar per source over the label span."""
    start, end = label_span(dataset_root, run_id)
    span = end - start
    if span <= 0:
        raise ValueError(f"{run_id}: label grid span is not positive")
    wanted = {(feature, source) for source, feature in sources.items()}
    counts: dict[str, int] = {source: 0 for source in sources}
    with (dataset_root / "telemetry" / f"{run_id}.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["feature"], row["source"])
            if key not in wanted:
                continue
            stamp = float(row["timestamp"])
            if start <= stamp <= end:
                counts[row["source"]] += 1
    return {source: count / span for source, count in counts.items()}


# ----------------------------------------------------------------------- statistics
def standardised_mean_difference(reference: np.ndarray, check: np.ndarray) -> float:
    if len(reference) == 0 or len(check) == 0:
        return math.nan
    mean_difference = float(check.mean() - reference.mean())
    if len(reference) < 2 and len(check) < 2:
        return 0.0 if mean_difference == 0.0 else math.inf
    variance_reference = float(reference.var(ddof=1)) if len(reference) > 1 else 0.0
    variance_check = float(check.var(ddof=1)) if len(check) > 1 else 0.0
    weight_reference = max(len(reference) - 1, 0)
    weight_check = max(len(check) - 1, 0)
    pooled = math.sqrt(
        (variance_reference * weight_reference + variance_check * weight_check)
        / max(weight_reference + weight_check, 1)
    )
    if pooled == 0.0:
        return 0.0 if abs(mean_difference) < 1e-12 else math.inf
    return mean_difference / pooled


def summary(values: np.ndarray, probs: list[float]) -> dict:
    if len(values) == 0:
        return {"n": 0, "mean": None, "sd": None, "quantiles": {}}
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "quantiles": {
            f"q{int(round(p * 100)):02d}": float(np.quantile(values, p)) for p in probs
        },
    }


def compare(reference: np.ndarray, check: np.ndarray, probs: list[float]) -> dict:
    reference_summary = summary(reference, probs)
    check_summary = summary(check, probs)
    return {
        "smd": _finite_or_string(standardised_mean_difference(reference, check)),
        "reference": reference_summary,
        "check": check_summary,
        "quantile_shift": {
            key: float(check_summary["quantiles"][key] - reference_summary["quantiles"][key])
            for key in check_summary["quantiles"]
            if key in reference_summary["quantiles"]
        },
    }


def _finite_or_string(value: float):
    if isinstance(value, float) and math.isinf(value):
        return "infinite"
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def _magnitude(value) -> float:
    if value == "infinite":
        return math.inf
    if value is None:
        return 0.0
    return abs(float(value))


def highlighted(column: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(column, pattern) for pattern in patterns)


# ------------------------------------------------------------------- feature shift
def feature_shift(
    reference_root: Path, reference_rows: list[dict], check_root: Path, check_rows: list[dict],
    columns: list[str], policy: dict,
) -> tuple[dict, list[str]]:
    probs = [float(p) for p in policy["quantiles"]]
    smd_policy = policy["standardised_mean_difference"]
    maximum = float(smd_policy["maximum_absolute"])
    patterns = list(smd_policy.get("highlighted_channels", []))
    minimum_episodes = int(policy.get("minimum_episodes_per_condition", 1))
    findings: list[str] = []
    report: dict = {}
    for condition in sorted(set(CONDITIONS.values())):
        reference_ids = [row["run_id"] for row in reference_rows if condition_of(row) == condition]
        check_ids = [row["run_id"] for row in check_rows if condition_of(row) == condition]
        if len(check_ids) < minimum_episodes:
            findings.append(
                f"{condition}: {len(check_ids)} check episodes, {minimum_episodes} required"
            )
        if not reference_ids:
            findings.append(f"{condition}: no reference episodes")
        pooled = {"reference": {c: [] for c in columns}, "check": {c: [] for c in columns}}
        episode_means = {"reference": {c: [] for c in columns}, "check": {c: [] for c in columns}}
        for side, root, run_ids in (
            ("reference", reference_root, reference_ids), ("check", check_root, check_ids),
        ):
            for run_id in run_ids:
                values = feature_rows(root, run_id, columns)
                for column in columns:
                    pooled[side][column].extend(values[column])
                    if values[column]:
                        episode_means[side][column].append(float(np.mean(values[column])))
        features: dict = {}
        exceeded: list[str] = []
        for column in columns:
            reference_values = np.asarray(pooled["reference"][column], dtype=float)
            check_values = np.asarray(pooled["check"][column], dtype=float)
            comparison = compare(reference_values, check_values, probs)
            comparison["episode_level_smd"] = _finite_or_string(standardised_mean_difference(
                np.asarray(episode_means["reference"][column], dtype=float),
                np.asarray(episode_means["check"][column], dtype=float),
            ))
            comparison["highlighted"] = highlighted(column, patterns)
            unit_value = (
                comparison["smd"] if smd_policy.get("unit", "decision") == "decision"
                else comparison["episode_level_smd"]
            )
            comparison["exceeds_maximum"] = _magnitude(unit_value) > maximum
            if comparison["exceeds_maximum"]:
                exceeded.append(column)
            if not comparison["highlighted"]:
                # keep the report compact: full quantiles only for highlighted channels
                comparison["reference"] = {"n": comparison["reference"]["n"], "mean": comparison["reference"]["mean"], "sd": comparison["reference"]["sd"]}
                comparison["check"] = {"n": comparison["check"]["n"], "mean": comparison["check"]["mean"], "sd": comparison["check"]["sd"]}
                comparison.pop("quantile_shift", None)
            features[column] = comparison
        if exceeded:
            findings.append(
                f"{condition}: deployable features shifted beyond {maximum} SMD: "
                + ", ".join(exceeded)
            )
        report[condition] = {
            "reference_episodes": len(reference_ids),
            "check_episodes": len(check_ids),
            "eligible_decisions": {
                "reference": int(len(pooled["reference"][columns[0]])) if columns else 0,
                "check": int(len(pooled["check"][columns[0]])) if columns else 0,
            },
            "maximum_absolute_smd": max((_magnitude(f["smd"]) for f in features.values()), default=0.0),
            "features_beyond_maximum": exceeded,
            "features": features,
        }
    return report, findings


def rate_shift(
    reference_root: Path, reference_rows: list[dict], check_root: Path, check_rows: list[dict],
    policy: dict,
) -> tuple[dict, list[str]]:
    rate_policy = policy["message_rate_proxies"]
    sources = dict(rate_policy["sources"])
    probs = [float(p) for p in policy["quantiles"]]
    maximum = float(rate_policy["maximum_absolute_smd"])
    report_only = bool(rate_policy.get("report_only", True))
    findings: list[str] = []
    report: dict = {}
    for condition in sorted(set(CONDITIONS.values())):
        sides = {}
        for side, root, rows in (("reference", reference_root, reference_rows), ("check", check_root, check_rows)):
            per_source: dict[str, list[float]] = {source: [] for source in sources}
            for row in rows:
                if condition_of(row) != condition:
                    continue
                rates = message_rates(root, row["run_id"], sources)
                for source, rate in rates.items():
                    per_source[source].append(rate)
            sides[side] = per_source
        entries = {}
        exceeded = []
        for source in sources:
            comparison = compare(
                np.asarray(sides["reference"][source], dtype=float),
                np.asarray(sides["check"][source], dtype=float), probs,
            )
            comparison["exceeds_maximum"] = _magnitude(comparison["smd"]) > maximum
            if comparison["exceeds_maximum"]:
                exceeded.append(source)
            entries[source] = comparison
        if exceeded and not report_only:
            findings.append(
                f"{condition}: message-rate proxies shifted beyond {maximum} SMD: " + ", ".join(exceeded)
            )
        report[condition] = {"sources_beyond_maximum": exceeded, "rates_hz": entries}
    return report, findings


# ------------------------------------------------------------------ false alerts
def clean_false_alerts(
    rows: list[dict], run_ids: set[str], alarm_policy: AlarmPolicy,
) -> dict:
    """False alerts per clean mission under a frozen policy for the given run ids."""
    selected = [row for row in rows if str(row["run_id"]) in run_ids and row.get("fault_family") == "none"]
    episodes = group_episodes(selected)
    missing = sorted(run_ids - set(episodes))
    predicted = {run_id: apply_alarm_policy(rows, alarm_policy) for run_id, rows in episodes.items()}
    if not predicted:
        return {"clean_missions": 0, "false_alerts": 0, "false_alerts_per_clean_mission": None,
                "missions_without_predictions": missing}
    metrics = evaluate_event_warnings(predicted)
    false_alerts = sum(int(item["false_alerts"]) for item in metrics["per_episode"])
    return {
        "clean_missions": len(predicted),
        "false_alerts": false_alerts,
        "false_alerts_per_clean_mission": false_alerts / len(predicted),
        "missions_without_predictions": missing,
        "per_mission": {
            item["run_id"]: int(item["false_alerts"]) for item in metrics["per_episode"]
        },
    }


def false_alert_shift(
    check_rows: list[dict], reference_rows: list[dict], *, check_predictions: Path,
    reference_predictions: Path, threshold_record: Path, alarm_config: dict, policy: dict,
) -> tuple[dict, list[str]]:
    settings = policy_settings(alarm_config)
    record = yaml.safe_load(threshold_record.read_text(encoding="utf-8"))
    threshold = float(record["threshold"])
    alarm_policy = AlarmPolicy(
        threshold, settings["required_above"], settings["decisions_considered"],
        settings["cooldown_seconds"],
    )
    budget = float(policy["false_alerts"]["budget_per_clean_mission"])
    findings: list[str] = []
    check_table = read_prediction_table(check_predictions)
    guard_protected_rows(check_table, explicitly_allowed=False)
    reference_table = (
        check_table if reference_predictions == check_predictions
        else read_prediction_table(reference_predictions)
    )
    guard_protected_rows(reference_table, explicitly_allowed=False)
    check_clean = {row["run_id"] for row in check_rows if condition_of(row) in CLEAN_CONDITIONS}
    reference_clean = {row["run_id"] for row in reference_rows if condition_of(row) in CLEAN_CONDITIONS}
    check = clean_false_alerts(check_table, check_clean, alarm_policy)
    reference = clean_false_alerts(reference_table, reference_clean, alarm_policy)
    by_condition = {}
    for condition in CLEAN_CONDITIONS:
        check_ids = {row["run_id"] for row in check_rows if condition_of(row) == condition}
        reference_ids = {row["run_id"] for row in reference_rows if condition_of(row) == condition}
        condition_check = clean_false_alerts(check_table, check_ids, alarm_policy)
        condition_reference = clean_false_alerts(reference_table, reference_ids, alarm_policy)
        condition_check.pop("per_mission", None)
        condition_reference.pop("per_mission", None)
        check_burden = condition_check["false_alerts_per_clean_mission"]
        reference_burden = condition_reference["false_alerts_per_clean_mission"]
        by_condition[condition] = {
            "check": condition_check,
            "sequential_reference": condition_reference,
            "difference_per_clean_mission": (
                None if check_burden is None or reference_burden is None
                else check_burden - reference_burden
            ),
            # diagnostic only: the pass rule is the pooled clean-mission budget
            "check_within_budget": check_burden is not None and check_burden <= budget,
        }
    if check["missions_without_predictions"]:
        findings.append(
            f"{len(check['missions_without_predictions'])} clean check missions have no predictions"
        )
    burden = check["false_alerts_per_clean_mission"]
    if burden is None:
        findings.append("no clean check missions were scored")
    elif burden > budget:
        findings.append(
            f"clean-mission false alerts per mission under parallel execution {burden:.4f} "
            f"exceed the {budget} budget"
        )
    model_ids = sorted({str(row["model_id"]) for row in check_table})
    return {
        "threshold": threshold,
        "threshold_record": str(threshold_record),
        "threshold_record_sha256": sha256_file(threshold_record),
        "model_id": model_ids,
        "persistence": {
            "required_above": settings["required_above"],
            "decisions_considered": settings["decisions_considered"],
        },
        "cooldown_seconds": settings["cooldown_seconds"],
        "budget_per_clean_mission": budget,
        "pass_rule_scope": "pooled_clean_missions",
        "check": check,
        "sequential_reference": reference,
        "by_condition": by_condition,
        "difference_per_clean_mission": (
            None if burden is None or reference["false_alerts_per_clean_mission"] is None
            else burden - reference["false_alerts_per_clean_mission"]
        ),
        "within_budget": burden is not None and burden <= budget,
    }, findings


# ------------------------------------------------------------------------ driver
def relative(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def evaluate(
    *, check_root: Path, reference_root: Path, policy: dict, schema: dict,
    check_predictions: Path | None = None, reference_predictions: Path | None = None,
    threshold_record: Path | None = None, alarm_config: dict | None = None,
) -> dict:
    check_rows = load_manifest_rows(check_root)
    reference_rows = load_manifest_rows(reference_root)
    columns = deployable_columns(schema)
    findings: list[str] = []
    features, feature_findings = feature_shift(
        reference_root, reference_rows, check_root, check_rows, columns, policy,
    )
    findings.extend(feature_findings)
    rates, rate_findings = rate_shift(reference_root, reference_rows, check_root, check_rows, policy)
    findings.extend(rate_findings)
    false_alerts = None
    supplied = check_predictions is not None or threshold_record is not None
    if supplied:
        if check_predictions is None or threshold_record is None or alarm_config is None:
            raise ValueError("false-alert evaluation needs predictions, a threshold record and the alarm policy")
        false_alerts, alert_findings = false_alert_shift(
            check_rows, reference_rows, check_predictions=check_predictions,
            reference_predictions=reference_predictions or check_predictions,
            threshold_record=threshold_record, alarm_config=alarm_config, policy=policy,
        )
        findings.extend(alert_findings)
    elif policy["false_alerts"].get("required_when_predictions_supplied", True):
        findings.append(
            "no frozen prediction table supplied: clean-mission false alerts under parallel "
            "execution were not evaluated"
        )
    return {
        "schema_version": 1,
        "check_campaign_id": check_campaign_id_of(check_root.name, policy),
        "check_dataset_id": check_root.name,
        "policy_check_campaign_id": policy.get("check_campaign_id"),
        "reference_dataset_id": reference_root.name,
        "comparison": policy.get("comparison", "condition_matched"),
        "decision_scope": policy.get("decision_scope", "eligible_decisions_only"),
        "counts": {
            "check_episodes": len(check_rows),
            "reference_episodes": len(reference_rows),
            "check_by_condition": {
                condition: sum(1 for row in check_rows if condition_of(row) == condition)
                for condition in sorted(set(CONDITIONS.values()))
            },
            "reference_by_condition": {
                condition: sum(1 for row in reference_rows if condition_of(row) == condition)
                for condition in sorted(set(CONDITIONS.values()))
            },
            "deployable_columns": len(columns),
        },
        "pass_rule": {
            "maximum_absolute_smd": float(policy["standardised_mean_difference"]["maximum_absolute"]),
            "smd_unit": policy["standardised_mean_difference"].get("unit", "decision"),
            "feature_scope": policy["standardised_mean_difference"].get("feature_scope"),
            "message_rate_report_only": bool(policy["message_rate_proxies"].get("report_only", True)),
            "false_alert_budget_per_clean_mission": float(policy["false_alerts"]["budget_per_clean_mission"]),
            "false_alerts_evaluated": false_alerts is not None,
        },
        "feature_shift": features,
        "message_rate_shift": rates,
        "false_alerts": false_alerts,
        "findings": findings,
        "passed": not findings,
        "protected_test_used": False,
        "training_performed": False,
        "interpretation": (
            "Engineering check of whether six concurrent simulators move the deployable "
            "health channels or the frozen policy's clean-mission false-alert burden relative "
            "to sequential collection. It admits nothing to the fitting pool by itself."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-dataset-id", required=True)
    parser.add_argument("--reference-dataset-id", default=None)
    parser.add_argument("--derived-root", type=Path, default=ROOT / "data/derived")
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/concurrency_shift_policy.yaml")
    parser.add_argument("--feature-schema", type=Path, default=ROOT / "configs/feature_schema.yaml")
    parser.add_argument("--predictions", type=Path, default=None,
                        help="frozen-model prediction table covering the check episodes")
    parser.add_argument("--reference-predictions", type=Path, default=None,
                        help="prediction table covering the sequential clean episodes "
                             "(defaults to --predictions)")
    parser.add_argument("--threshold-record", type=Path, default=None)
    parser.add_argument("--alarm", type=Path, default=ROOT / "configs/alarm_policy.yaml")
    parser.add_argument("--output", type=Path, default=None,
                        help="defaults to reports/integrity/<check campaign id>.yaml")
    args = parser.parse_args()
    policy = yaml.safe_load(args.policy.read_text(encoding="utf-8"))
    campaign_id = check_campaign_id_of(args.check_dataset_id, policy)
    if args.output is None:
        if not campaign_id:
            raise SystemExit("--output is required when the check dataset id names no campaign")
        args.output = ROOT / "reports/integrity" / f"{campaign_id}.yaml"
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable report {args.output}")
    schema = yaml.safe_load(args.feature_schema.read_text(encoding="utf-8"))
    reference_id = args.reference_dataset_id or str(policy["reference_dataset_id"])
    check_root = args.derived_root / args.check_dataset_id
    reference_root = args.derived_root / reference_id
    alarm_config = yaml.safe_load(args.alarm.read_text(encoding="utf-8")) if args.predictions else None
    report = evaluate(
        check_root=check_root, reference_root=reference_root, policy=policy, schema=schema,
        check_predictions=args.predictions, reference_predictions=args.reference_predictions,
        threshold_record=args.threshold_record, alarm_config=alarm_config,
    )
    inputs = {
        "policy": relative(args.policy), "policy_sha256": sha256_file(args.policy),
        "feature_schema_sha256": sha256_file(args.feature_schema),
        "check_extraction_manifest_sha256": sha256_file(check_root / "extraction_manifest.jsonl"),
        "reference_extraction_manifest_sha256": sha256_file(reference_root / "extraction_manifest.jsonl"),
    }
    if args.predictions:
        inputs["predictions"] = relative(args.predictions)
        inputs["predictions_sha256"] = sha256_file(args.predictions)
        if args.reference_predictions:
            inputs["reference_predictions"] = relative(args.reference_predictions)
            inputs["reference_predictions_sha256"] = sha256_file(args.reference_predictions)
        inputs["alarm_policy_sha256"] = sha256_file(args.alarm)
    report = {**report, "inputs": inputs}
    write_yaml_report(args.output, report)
    print(json.dumps({
        "passed": report["passed"], "findings": report["findings"], "output": str(args.output),
    }, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

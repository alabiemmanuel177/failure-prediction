#!/usr/bin/env python3
"""One-time confirmatory evaluation on the protected held-out maps.

Consumes immutable, calibrated, alarm-applied prediction tables (contract columns plus
``alarm``/``persistent``) for P3 and P1 over the SAME held-out episodes, optionally
P2/P4/P5/P6 (exploratory), and the frozen threshold from ``configs/alarm_policy.yaml``.

Refuses before the model freeze: protected tables pass
``src.protected_data.enforce_protected_boundary`` only with
``--allow-protected-after-freeze`` and a passing ``check_readiness --stage confirmatory``.

Computes event recall, false alerts per clean and per non-event mission, the lead-time
distribution with its full denominator (undetected events reported), all of it by
family, severity, map and route; H1 (paired P3 minus P1 event recall with a
map -> route -> episode bootstrap interval), H2 (median lead time >= 3 s), H3
(before/after calibration Brier and ECE, validation versus held-out), H4 (ablation
deltas from ``reports/ablations`` when present). Mission timeouts are analysed
separately. Writes the immutable ``reports/confirmatory/held_out_map.yaml``.

``--engineering-fixture`` runs on synthetic tables for tests only and marks the output
as non-research evidence (``complete: false``).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes
from src.evaluation import brier_score, evaluate_event_warnings, reliability_curve
from src.protected_data import HELD_OUT_SPLIT, enforce_protected_boundary

CONTRACT_COLUMNS = (
    "run_id", "decision_index", "decision_time", "split", "map_id", "route_id",
    "fault_family", "severity", "seed", "protected_test_used", "eligibility", "label",
    "primary_event_class", "primary_event_time", "model_id", "raw_score", "risk_score",
)
ALARM_COLUMN = "alarm"
TIMEOUT_CLASS = "mission_timeout"
LEAD_THRESHOLDS = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0)
H2_MINIMUM_LEAD_SECONDS = 3.0
EXPLORATORY_MODELS = ("p2", "p4", "p5", "p6")
H4_GROUPS = ("no_planner_controller", "no_localisation")


def truth(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(values: Sequence[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def percentile(values: Sequence[float], probability: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


# --------------------------------------------------------------------------- tables

def load_table(path: Path, *, require_alarm: bool = True) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        rows = list(reader)
    missing = [column for column in CONTRACT_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"{path}: missing contract columns {missing}")
    if require_alarm and ALARM_COLUMN not in columns:
        raise ValueError(
            f"{path}: missing '{ALARM_COLUMN}' column; apply scripts/apply_alarm_policy.py "
            "with the frozen policy before confirmatory evaluation"
        )
    if not rows:
        raise ValueError(f"{path}: prediction table is empty")
    return rows


def group_episodes(rows: Sequence[Mapping[str, str]]) -> dict[str, list[dict[str, str]]]:
    episodes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        episodes[row["run_id"]].append(dict(row))
    for run_id, episode_rows in episodes.items():
        episode_rows.sort(key=lambda row: float(row["decision_time"]))
        times = [float(row["decision_time"]) for row in episode_rows]
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise ValueError(f"{run_id}: decision times are not strictly increasing")
    return dict(episodes)


def table_findings(
    rows: Sequence[Mapping[str, str]], *, expected_split: str, expected_protected: bool,
) -> list[str]:
    findings: list[str] = []
    splits = {row["split"] for row in rows}
    if splits != {expected_split}:
        findings.append(f"table split values {sorted(splits)} differ from {expected_split!r}")
    protected = {truth(row["protected_test_used"]) for row in rows}
    if protected != {expected_protected}:
        findings.append(f"protected_test_used values {sorted(protected)} differ from {expected_protected}")
    models = {row["model_id"] for row in rows}
    if len(models) != 1:
        findings.append(f"table mixes model ids {sorted(models)}")
    return findings


def episode_metadata(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    first = rows[0]
    classes = {row["primary_event_class"] for row in rows if row["primary_event_class"]}
    if len(classes) > 1:
        raise ValueError(f"{first['run_id']}: inconsistent primary_event_class")
    return {
        "map_id": first["map_id"],
        "route_id": first["route_id"],
        "fault_family": first["fault_family"],
        "severity": first["severity"],
        "seed": first["seed"],
        "primary_event_class": next(iter(classes)) if classes else None,
    }


def episode_outcomes(rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, object]]:
    """Per-episode detection outcomes at the frozen, already-applied alarm policy."""
    episodes = group_episodes(rows)
    metrics = evaluate_event_warnings(episodes)
    outcomes: dict[str, dict[str, object]] = {}
    for item in metrics["per_episode"]:
        run_id = item["run_id"]
        meta = episode_metadata(episodes[run_id])
        outcomes[run_id] = {
            "run_id": run_id,
            **meta,
            "has_event": bool(item["has_event"]),
            "is_timeout": meta["primary_event_class"] == TIMEOUT_CLASS,
            "detected": bool(item["detected"]),
            "lead_seconds": item["first_useful_lead_seconds"],
            "false_alerts": int(item["false_alerts"]),
            "is_clean": meta["fault_family"] == "none",
        }
    return outcomes


# ------------------------------------------------------------------------ bootstrap

def hierarchical_bootstrap(
    records: Sequence[Mapping[str, object]],
    statistic: Callable[[list[Mapping[str, object]]], float | None],
    *,
    replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260903,
) -> dict[str, object]:
    """Map -> route -> episode resampling of independent episodes (paired by record).

    Mirrors the resampling structure of ``src.evaluation.bootstrap`` but accepts any
    per-sample statistic so paired differences, recalls and medians share one design.
    """
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not records:
        raise ValueError("no records to bootstrap")
    identities = [row["run_id"] for row in records]
    if len(identities) != len(set(identities)):
        raise ValueError("bootstrap records must be unique by episode")
    hierarchy: dict[object, dict[object, list[Mapping[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in records:
        hierarchy[row["map_id"]][row["route_id"]].append(row)
    maps = sorted(hierarchy, key=str)
    point = statistic(list(records))
    rng = random.Random(seed)
    draws: list[float] = []
    undefined = 0
    for _ in range(replicates):
        sample: list[Mapping[str, object]] = []
        for _map_draw in range(len(maps)):
            map_id = rng.choice(maps)
            routes = sorted(hierarchy[map_id], key=str)
            for _route_draw in range(len(routes)):
                route_id = rng.choice(routes)
                episodes = hierarchy[map_id][route_id]
                for _episode_draw in range(len(episodes)):
                    sample.append(rng.choice(episodes))
        value = statistic(sample)
        if value is None:
            undefined += 1
        else:
            draws.append(value)
    alpha = (1.0 - confidence) / 2.0
    interval = (
        [percentile(draws, alpha), percentile(draws, 1.0 - alpha)] if draws else [None, None]
    )
    return {
        "analysis_unit": "episode",
        "hierarchy": ["map", "route", "episode"],
        "episode_count": len(records),
        "map_count": len(maps),
        "route_count": sum(len(routes) for routes in hierarchy.values()),
        "point_estimate": point,
        "confidence": confidence,
        "confidence_interval": interval,
        "bootstrap_replicates": replicates,
        "undefined_replicates": undefined,
        "seed": seed,
    }


def recall_statistic(rows: list[Mapping[str, object]]) -> float | None:
    events = [row for row in rows if row["has_event"]]
    if not events:
        return None
    return sum(1 for row in events if row["detected"]) / len(events)


def paired_recall_difference(rows: list[Mapping[str, object]]) -> float | None:
    events = [row for row in rows if row["has_event"]]
    if not events:
        return None
    return sum(float(row["detected_a"]) - float(row["detected_b"]) for row in events) / len(events)


def median_lead_statistic(rows: list[Mapping[str, object]]) -> float | None:
    leads = [float(row["lead_seconds"]) for row in rows if row["detected"] and row["lead_seconds"] is not None]
    return median(leads)


# -------------------------------------------------------------------------- metrics

def lead_time_distribution(events: Sequence[Mapping[str, object]]) -> dict[str, object]:
    leads = [float(row["lead_seconds"]) for row in events if row["detected"]]
    total = len(events)
    return {
        "event_count_full_denominator": total,
        "detected_event_count": len(leads),
        "undetected_event_count": total - len(leads),
        "median_seconds_detected": median(leads),
        "quantiles_seconds_detected": {
            "p10": percentile(leads, 0.10), "p25": percentile(leads, 0.25),
            "p50": percentile(leads, 0.50), "p75": percentile(leads, 0.75),
            "p90": percentile(leads, 0.90),
        },
        "minimum_seconds_detected": min(leads) if leads else None,
        "maximum_seconds_detected": max(leads) if leads else None,
        "warned_fraction_of_all_events_by_lead": {
            f"{threshold:g}s": (sum(1 for lead in leads if lead >= threshold) / total) if total else None
            for threshold in LEAD_THRESHOLDS
        },
    }


def summarize(outcomes: Sequence[Mapping[str, object]]) -> dict[str, object]:
    events = [row for row in outcomes if row["has_event"]]
    non_events = [row for row in outcomes if not row["has_event"]]
    clean = [row for row in outcomes if row["is_clean"]]
    detected = sum(1 for row in events if row["detected"])
    return {
        "episode_count": len(outcomes),
        "event_count": len(events),
        "detected_event_count": detected,
        "undetected_event_count": len(events) - detected,
        "event_recall": detected / len(events) if events else None,
        "non_event_mission_count": len(non_events),
        "false_alerts_per_non_event_mission": (
            sum(row["false_alerts"] for row in non_events) / len(non_events) if non_events else None
        ),
        "clean_mission_count": len(clean),
        "false_alerts_per_clean_mission": (
            sum(row["false_alerts"] for row in clean) / len(clean) if clean else None
        ),
        "false_alert_count_all_missions": sum(row["false_alerts"] for row in outcomes),
        "lead_time": lead_time_distribution(events),
    }


def stratified(outcomes: Sequence[Mapping[str, object]], field: str) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in outcomes:
        groups[str(row[field])].append(row)
    return {key: summarize(rows) for key, rows in sorted(groups.items())}


def model_report(outcomes: Mapping[str, Mapping[str, object]], *, bootstrap: dict) -> dict[str, object]:
    primary = [row for row in outcomes.values() if not row["is_timeout"]]
    events = [row for row in primary if row["has_event"]]
    report = {
        "primary_events_exclude_mission_timeout": True,
        **summarize(primary),
        "by_family": stratified(primary, "fault_family"),
        "by_severity": stratified(primary, "severity"),
        "by_map": stratified(primary, "map_id"),
        "by_route": stratified(primary, "route_id"),
    }
    if events:
        report["event_recall_bootstrap"] = hierarchical_bootstrap(events, recall_statistic, **bootstrap)
    return report


def timeout_report(models: Mapping[str, Mapping[str, Mapping[str, object]]]) -> dict[str, object]:
    result: dict[str, object] = {"analysed_separately": True, "event_class": TIMEOUT_CLASS}
    for model_id, outcomes in models.items():
        timeouts = [row for row in outcomes.values() if row["is_timeout"]]
        everything = list(outcomes.values())
        result[model_id] = {
            "timeout_episode_count": len(timeouts),
            "timeout_recall": (
                sum(1 for row in timeouts if row["detected"]) / len(timeouts) if timeouts else None
            ),
            "lead_time_timeouts": lead_time_distribution(timeouts),
            "all_events_including_timeouts": summarize(everything),
        }
    return result


# ---------------------------------------------------------------------- hypotheses

def pair(p3: Mapping[str, Mapping[str, object]], p1: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    if set(p3) != set(p1):
        raise ValueError("P3 and P1 tables must cover the same held-out episodes")
    paired = []
    for run_id in sorted(p3):
        a, b = p3[run_id], p1[run_id]
        if a["has_event"] != b["has_event"] or a["primary_event_class"] != b["primary_event_class"]:
            raise ValueError(f"{run_id}: label columns differ between P3 and P1 tables")
        paired.append({**a, "detected_a": a["detected"], "detected_b": b["detected"]})
    return paired


def h1_report(paired: Sequence[Mapping[str, object]], *, bootstrap: dict) -> dict[str, object]:
    events = [row for row in paired if row["has_event"] and not row["is_timeout"]]
    if not events:
        return {"status": "not_evaluable", "reason": "no primary events on held-out episodes"}
    result = hierarchical_bootstrap(events, paired_recall_difference, **bootstrap)
    lower = result["confidence_interval"][0]
    both = sum(1 for row in events if row["detected_a"] and row["detected_b"])
    only_a = sum(1 for row in events if row["detected_a"] and not row["detected_b"])
    only_b = sum(1 for row in events if row["detected_b"] and not row["detected_a"])
    neither = len(events) - both - only_a - only_b
    return {
        "label": "confirmatory",
        "hypothesis": "P3 event recall exceeds P1 at the validation-fixed false-alert budget",
        "contrast": "p3_recall_minus_p1_recall",
        "denominator_events": len(events),
        "p3_recall": sum(1 for row in events if row["detected_a"]) / len(events),
        "p1_recall": sum(1 for row in events if row["detected_b"]) / len(events),
        "discordant_pairs": {"p3_only": only_a, "p1_only": only_b, "both": both, "neither": neither},
        "supported": bool(lower is not None and lower > 0.0),
        "criterion": "95% hierarchical bootstrap interval of the paired difference excludes zero from below",
        **result,
    }


def h2_report(p3: Mapping[str, Mapping[str, object]], *, bootstrap: dict) -> dict[str, object]:
    events = [row for row in p3.values() if row["has_event"] and not row["is_timeout"]]
    detected = [row for row in events if row["detected"]]
    if not detected:
        return {"label": "supporting", "status": "not_evaluable", "reason": "no detected events",
                "denominator_events": len(events)}
    result = hierarchical_bootstrap(events, median_lead_statistic, **bootstrap)
    lower = result["confidence_interval"][0]
    return {
        "label": "supporting",
        "hypothesis": f"median useful lead time of detected failures >= {H2_MINIMUM_LEAD_SECONDS:g} s",
        "denominator_events": len(events),
        "detected_events": len(detected),
        "undetected_events": len(events) - len(detected),
        "median_lead_seconds": result["point_estimate"],
        "point_estimate_meets_minimum": bool(result["point_estimate"] >= H2_MINIMUM_LEAD_SECONDS),
        "interval_lower_bound_meets_minimum": bool(lower is not None and lower >= H2_MINIMUM_LEAD_SECONDS),
        "warned_fraction_of_all_events_at_least_3s": (
            sum(1 for row in detected if float(row["lead_seconds"]) >= H2_MINIMUM_LEAD_SECONDS) / len(events)
        ),
        **result,
    }


def calibration_pair(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    before = [{**row, "risk_score": row["raw_score"]} for row in rows]
    curve_before = reliability_curve(before)
    curve_after = reliability_curve(rows)
    return {
        "eligible_decision_count": curve_after["eligible_decision_count"],
        "brier_before": brier_score(before),
        "brier_after": curve_after["brier_score"],
        "ece_before": curve_before["ece"],
        "ece_after": curve_after["ece"],
        "brier_delta_after_minus_before": curve_after["brier_score"] - brier_score(before),
        "ece_delta_after_minus_before": curve_after["ece"] - curve_before["ece"],
        "reliability_bins_after": curve_after["bins"],
    }


def h3_report(validation_rows, held_out_rows) -> dict[str, object]:
    held_out = calibration_pair(held_out_rows)
    report = {
        "label": "supporting",
        "hypothesis": "calibration reduces Brier score and ECE without reducing event recall beyond tolerance",
        "before": "raw_score (uncalibrated sigmoid output)",
        "after": "risk_score (frozen calibrator applied)",
        "held_out": held_out,
        "held_out_brier_reduced": held_out["brier_delta_after_minus_before"] < 0,
        "held_out_ece_reduced": held_out["ece_delta_after_minus_before"] < 0,
        "recall_tolerance_note": (
            "event recall at the frozen threshold is the primary P3 result; the declared "
            "tolerance is assessed against the validation selection record, not here"
        ),
    }
    if validation_rows is None:
        report["validation"] = {"status": "not_provided"}
        report["supported"] = None
    else:
        validation = calibration_pair(validation_rows)
        report["validation"] = validation
        report["supported"] = bool(
            validation["brier_delta_after_minus_before"] < 0
            and validation["ece_delta_after_minus_before"] < 0
        )
    return report


def h4_report(ablation_dir: Path | None, primary_recall: float | None) -> dict[str, object]:
    report: dict[str, object] = {
        "label": "supporting",
        "hypothesis": "removing planner and localisation health features materially reduces early warning",
        "primary_event_recall": primary_recall,
        "ablations": {},
        "material_decline_threshold": "not prespecified numerically; deltas and intervals reported",
    }
    if ablation_dir is None or not ablation_dir.is_dir():
        report["status"] = "ablation_reports_absent"
        return report
    for path in sorted(ablation_dir.glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(document, dict):
            continue
        name = document.get("ablation") or document.get("ablation_id") or path.stem
        metrics = document.get("metrics") if isinstance(document.get("metrics"), dict) else document
        recall = metrics.get("event_recall")
        entry = {
            "source": str(path),
            "source_sha256": sha256_file(path),
            "event_recall": recall,
            "protected_test_used": document.get("protected_test_used"),
        }
        if isinstance(recall, (int, float)) and primary_recall is not None:
            entry["recall_delta_vs_primary"] = float(recall) - float(primary_recall)
        if "confidence_interval" in metrics:
            entry["confidence_interval"] = metrics["confidence_interval"]
        report["ablations"][str(name)] = entry
    h4_entries = {name: report["ablations"][name] for name in H4_GROUPS if name in report["ablations"]}
    report["h4_groups"] = h4_entries
    report["status"] = "evaluated" if h4_entries else "h4_groups_not_found"
    return report


def exploratory_report(
    models: Mapping[str, Mapping[str, Mapping[str, object]]], p1: Mapping[str, Mapping[str, object]],
    *, bootstrap: dict,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for model_key, outcomes in models.items():
        paired = pair(outcomes, p1)
        events = [row for row in paired if row["has_event"] and not row["is_timeout"]]
        entry = {"label": "exploratory", "not_a_confirmatory_claim": True}
        if events:
            entry.update(hierarchical_bootstrap(events, paired_recall_difference, **bootstrap))
            entry["contrast"] = f"{model_key}_recall_minus_p1_recall"
        result[model_key] = entry
    return result


def build_report(
    *, tables: Mapping[str, Sequence[Mapping[str, str]]], threshold: float | None,
    alarm_policy: Mapping[str, object], validation_rows=None, ablation_dir: Path | None = None,
    bootstrap: dict | None = None, engineering_fixture: bool = False,
    inputs: Mapping[str, str] | None = None, model_freeze_sha256: str | None = None,
) -> dict[str, object]:
    bootstrap = bootstrap or {}
    outcomes = {key: episode_outcomes(rows) for key, rows in tables.items()}
    p3, p1 = outcomes["p3"], outcomes["p1"]
    paired = pair(p3, p1)
    models = {key: model_report(value, bootstrap=bootstrap) for key, value in outcomes.items()}
    exploratory = {key: outcomes[key] for key in EXPLORATORY_MODELS if key in outcomes}
    persistence = alarm_policy.get("persistence", {})
    return {
        "schema_version": 1,
        "protocol_version": alarm_policy.get("protocol_version"),
        "kind": "held_out_map_confirmatory_evaluation",
        "complete": not engineering_fixture,
        "research_evidence": not engineering_fixture,
        "engineering_fixture": engineering_fixture,
        "evidence_status": (
            "non_research_engineering_fixture" if engineering_fixture else "confirmatory_one_time"
        ),
        "protected_test_used": True,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_unit": "episode",
        "split": HELD_OUT_SPLIT,
        "frozen_threshold": threshold,
        "alarm_policy": {
            "threshold": threshold,
            "persistence": dict(persistence),
            "cooldown_seconds": alarm_policy.get("cooldown_seconds"),
            "false_alert_budget_per_clean_mission": alarm_policy.get("false_alert_budget_per_clean_mission"),
            "test_threshold_adaptation": "forbidden",
        },
        "model_freeze_sha256": model_freeze_sha256,
        "inputs_sha256": dict(inputs or {}),
        "model_ids": {key: rows[0]["model_id"] for key, rows in tables.items()},
        "episode_count": len(p3),
        "event_time_convention": (
            "lead = primary_event_time - decision_time of the first alarm on an "
            "eligible_positive decision; alarms outside the warning window count as false alerts"
        ),
        "models": models,
        "h1": h1_report(paired, bootstrap=bootstrap),
        "h2": h2_report(p3, bootstrap=bootstrap),
        "h3": h3_report(validation_rows, tables["p3"]),
        "h4": h4_report(ablation_dir, models["p3"]["event_recall"]),
        "timeouts": timeout_report(outcomes),
        "exploratory_models": exploratory_report(exploratory, p1, bootstrap=bootstrap),
    }


# ----------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p3", type=Path, required=True, help="calibrated, alarm-applied P3 held-out table")
    parser.add_argument("--p1", type=Path, required=True, help="alarm-applied P1 held-out table")
    for key in EXPLORATORY_MODELS:
        parser.add_argument(f"--{key}", type=Path, help=f"optional alarm-applied {key.upper()} table (exploratory)")
    parser.add_argument("--validation-p3", type=Path,
                        help="calibrated P3 validation table (raw_score vs risk_score) for H3")
    parser.add_argument("--ablations-dir", type=Path, default=ROOT / "reports/ablations")
    parser.add_argument("--alarm", type=Path, default=ROOT / "configs/alarm_policy.yaml")
    parser.add_argument("--model-freeze", type=Path, default=ROOT / "configs/model_freeze.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/confirmatory/held_out_map.yaml")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--allow-protected-after-freeze", action="store_true")
    parser.add_argument("--engineering-fixture", action="store_true",
                        help="synthetic tables only; output is marked non-research evidence")
    args = parser.parse_args(argv)

    alarm = yaml.safe_load(args.alarm.read_text(encoding="utf-8")) or {}
    threshold = alarm.get("threshold")
    if threshold is None and not args.engineering_fixture:
        raise SystemExit("configs/alarm_policy.yaml threshold is not frozen")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable report {args.output}")

    table_paths = {"p3": args.p3, "p1": args.p1}
    for key in EXPLORATORY_MODELS:
        if getattr(args, key) is not None:
            table_paths[key] = getattr(args, key)
    tables = {key: load_table(path) for key, path in table_paths.items()}

    if args.engineering_fixture:
        gate_passed = False
        model_freeze_sha256 = None
    else:
        gate = subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_readiness.py"), "--stage", "confirmatory"],
            check=False, capture_output=True, text=True,
        )
        gate_passed = gate.returncode == 0
        if not gate_passed:
            print(gate.stdout)
        try:
            for key, rows in tables.items():
                protected = {truth(row["protected_test_used"]) for row in rows}
                enforce_protected_boundary(
                    True if protected == {True} else (False if protected == {False} else None),
                    explicitly_allowed=args.allow_protected_after_freeze,
                    confirmatory_gate_passed=gate_passed,
                )
        except ValueError as error:
            raise SystemExit(f"confirmatory evaluation refused: {error}") from error
        model_freeze_sha256 = sha256_file(args.model_freeze)
        findings = []
        for key, rows in tables.items():
            findings.extend(f"{key}: {item}" for item in table_findings(
                rows, expected_split=HELD_OUT_SPLIT, expected_protected=True,
            ))
        if findings:
            raise SystemExit("invalid held-out tables:\n- " + "\n- ".join(findings))

    validation_rows = None
    if args.validation_p3 is not None:
        validation_rows = load_table(args.validation_p3, require_alarm=False)
        if not args.engineering_fixture:
            findings = table_findings(validation_rows, expected_split="validation", expected_protected=False)
            if findings:
                raise SystemExit("invalid validation table:\n- " + "\n- ".join(findings))

    inputs = {str(path): sha256_file(path) for path in table_paths.values()}
    if args.validation_p3 is not None:
        inputs[str(args.validation_p3)] = sha256_file(args.validation_p3)
    inputs[str(args.alarm)] = sha256_file(args.alarm)
    report = build_report(
        tables=tables, threshold=threshold, alarm_policy=alarm, validation_rows=validation_rows,
        ablation_dir=args.ablations_dir,
        bootstrap={"replicates": args.replicates, "seed": args.seed},
        engineering_fixture=args.engineering_fixture, inputs=inputs,
        model_freeze_sha256=model_freeze_sha256,
    )
    publish_new_bytes(args.output, yaml.safe_dump(report, sort_keys=False).encode("utf-8"))
    h1 = report["h1"]
    print(
        f"wrote {args.output}: H1 P3-P1 recall difference "
        f"{h1.get('point_estimate')} CI {h1.get('confidence_interval')} "
        f"(complete={report['complete']}, research_evidence={report['research_evidence']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

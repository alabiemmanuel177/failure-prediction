#!/usr/bin/env python3
"""Regenerate the results tables (Markdown + CSV) from the same immutable artifacts.

Outputs go to ``reports/tables/`` with one ``<table>.json`` sidecar per table recording
source artifacts and sha256. Missing inputs are listed as PENDING; nothing is invented.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
from typing import Any, Callable



ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reporting import (  # noqa: E402
    ArtifactRegistry, predictor_metrics, reliability_by_score, split_models,
)
from src.reporting.inputs import publish_text, write_sidecar  # noqa: E402


class Pending(Exception):
    pass


TABLES = (
    "tab01_predictor_summary", "tab02_recall_by_family", "tab03_unseen_family",
    "tab04_ablations", "tab05_recovery_outcomes", "tab06_action_confusion", "tab07_latency",
)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render(columns: list[str], rows: list[dict[str, Any]]) -> tuple[str, str]:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: ("" if row.get(column) is None else row.get(column)) for column in columns})
    header = "| " + " | ".join(columns) + " |\n|" + "|".join("---" for _ in columns) + "|\n"
    body = "".join("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |\n" for row in rows)
    return buffer.getvalue(), header + body


def _require(registry: ArtifactRegistry, *names: str) -> None:
    missing = [name for name in names if not registry.available(name)]
    if missing:
        raise Pending(", ".join(f"{name} ({registry.path(name)})" for name in missing))


def _summary_rows(registry: ArtifactRegistry, name: str, split_label: str) -> list[dict[str, Any]]:
    rows = registry.prediction_rows(name)
    output = []
    for model, model_rows in split_models(rows).items():
        metrics = predictor_metrics(model_rows)
        raw = reliability_by_score(model_rows, "raw_score")
        calibrated = reliability_by_score(model_rows, "risk_score")
        output.append({
            "split": split_label, "model_id": model,
            "episodes": metrics["episode_count"], "events": metrics["event_count"],
            "detected_events": metrics["detected_event_count"],
            "event_recall": metrics["event_recall"],
            "false_alerts_per_clean_mission": metrics["false_alerts_per_clean_mission"],
            "false_alerts_per_mission": metrics["false_alerts_per_mission"],
            "median_useful_lead_seconds": metrics["median_useful_lead_seconds_detected"],
            "brier_raw": raw["brier_score"], "ece_raw": raw["ece"],
            "brier_calibrated": calibrated["brier_score"], "ece_calibrated": calibrated["ece"],
        })
    return output


def tab01(registry: ArtifactRegistry) -> tuple[list[str], list[dict[str, Any]]]:
    _require(registry, "predictions_held_out")
    rows = _summary_rows(registry, "predictions_held_out", "held_out_map_test")
    if registry.available("predictions_validation"):
        rows = _summary_rows(registry, "predictions_validation", "validation") + rows
    columns = list(rows[0].keys())
    return columns, rows


def tab02(registry: ArtifactRegistry) -> tuple[list[str], list[dict[str, Any]]]:
    _require(registry, "predictions_held_out")
    rows = registry.prediction_rows("predictions_held_out")
    output = []
    for model, model_rows in split_models(rows).items():
        for family, entry in predictor_metrics(model_rows)["by_family"].items():
            output.append({"model_id": model, "fault_family": family, "episodes": entry["episodes"],
                           "events": entry["events"], "detected": entry["detected"],
                           "event_recall": entry["event_recall"],
                           "false_alerts_per_mission": entry["false_alerts_per_mission"]})
    return list(output[0].keys()), output


def tab03(registry: ArtifactRegistry) -> tuple[list[str], list[dict[str, Any]]]:
    _require(registry, "predictions_unseen_family")
    rows = registry.prediction_rows("predictions_unseen_family")
    if "fold_family" in rows[0]:
        rows = [row for row in rows if row["fold_family"] == row["fault_family"]]
    output = []
    for model, model_rows in split_models(rows).items():
        for family, entry in predictor_metrics(model_rows)["by_family"].items():
            if family == "none":
                continue
            output.append({"model_id": model, "excluded_family": family, "episodes": entry["episodes"],
                           "events": entry["events"], "detected": entry["detected"],
                           "event_recall": entry["event_recall"],
                           "false_alerts_per_mission": entry["false_alerts_per_mission"]})
    if not output:
        raise Pending("unseen-family table has no excluded-family rows")
    return list(output[0].keys()), output


def tab04(registry: ArtifactRegistry) -> tuple[list[str], list[dict[str, Any]]]:
    _require(registry, "predictions_ablation")
    rows = registry.prediction_rows("predictions_ablation")
    if "ablation" not in rows[0]:
        raise Pending("ablation prediction table lacks an 'ablation' column")
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(str(row["ablation"]), []).append(row)
    output = []
    for name in sorted(groups):
        metrics = predictor_metrics(groups[name])
        output.append({"ablation": name, "episodes": metrics["episode_count"], "events": metrics["event_count"],
                       "event_recall": metrics["event_recall"],
                       "false_alerts_per_clean_mission": metrics["false_alerts_per_clean_mission"],
                       "median_useful_lead_seconds": metrics["median_useful_lead_seconds_detected"]})
    return list(output[0].keys()), output


def tab05(registry: ArtifactRegistry) -> tuple[list[str], list[dict[str, Any]]]:
    _require(registry, "paired_recovery")
    report = registry.yaml("paired_recovery")
    comparisons = {k: v for k, v in report.get("comparisons", {}).items() if v.get("status") == "computed"}
    if not comparisons:
        raise Pending("paired recovery report has no computed comparison")
    output = []
    for name, item in sorted(comparisons.items()):
        for role in ("baseline", "proposed"):
            summary = item[role]
            output.append({
                "comparison": name, "policy": item[f"{role}_policy"], "episodes": summary["episode_count"],
                "mission_completion_rate": summary["mission_completion_rate"],
                "collision_rate": summary["collision_rate"],
                "guard_violations": summary["guard_violation_count"],
                "guard_rejections": summary["guard_rejection_count"],
                "median_added_time_seconds": summary["median_added_time_seconds"],
                "median_added_path_length_m": summary["median_added_path_length_m"],
                "mean_intervention_count": summary["mean_intervention_count"],
                "completion_difference": item["paired_completion_rate_difference"] if role == "proposed" else None,
                "completion_ci_low": item["completion_difference_bootstrap"]["confidence_interval"][0] if role == "proposed" else None,
                "completion_ci_high": item["completion_difference_bootstrap"]["confidence_interval"][1] if role == "proposed" else None,
                "estimator": item["mixed_effects_model"]["method"] if role == "proposed" else None,
            })
    return list(output[0].keys()), output


def tab06(registry: ArtifactRegistry) -> tuple[list[str], list[dict[str, Any]]]:
    _require(registry, "paired_recovery")
    report = registry.yaml("paired_recovery")
    confusion = report.get("action_confusion_vs_oracle", {})
    if "R3" not in confusion:
        raise Pending("paired recovery report lacks an oracle action confusion")
    output = []
    for policy, cells in sorted(confusion.items()):
        for key, count in sorted(cells.items()):
            oracle, chosen = key.split("->")
            output.append({"policy": policy, "oracle_action": oracle, "executed_action": chosen, "count": count})
    return ["policy", "oracle_action", "executed_action", "count"], output


def tab07(registry: ArtifactRegistry) -> tuple[list[str], list[dict[str, Any]]]:
    _require(registry, "latency_report")
    output = registry.latency_records()
    if not output:
        raise Pending("latency report has no model records")
    return ["model_id", "median_ms", "p95_ms", "max_ms"], output


BUILDERS: dict[str, Callable[[ArtifactRegistry], tuple[list[str], list[dict[str, Any]]]]] = {
    "tab01_predictor_summary": tab01, "tab02_recall_by_family": tab02, "tab03_unseen_family": tab03,
    "tab04_ablations": tab04, "tab05_recovery_outcomes": tab05, "tab06_action_confusion": tab06,
    "tab07_latency": tab07,
}


def build_all(registry: ArtifactRegistry, out: Path, only: list[str] | None = None) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    index: dict[str, Any] = {"generated_utc": generated_utc, "produced": {}, "pending": {}}
    for name in TABLES:
        if only and name not in only:
            continue
        registry.used.clear()
        try:
            columns, rows = BUILDERS[name](registry)
        except Pending as pending:
            index["pending"][name] = str(pending)
            continue
        csv_text, markdown = render(columns, rows)
        csv_path, md_path = out / f"{name}.csv", out / f"{name}.md"
        publish_text(csv_path, csv_text)
        publish_text(md_path, f"# {name}\n\nGenerated {generated_utc} from immutable artifacts.\n\n{markdown}")
        write_sidecar(csv_path, {"table": name, "generated_utc": generated_utc, "rows": len(rows),
                                 "sources": registry.sources(list(registry.used)), "fabricated_inputs": False})
        index["produced"][name] = [csv_path.name, md_path.name]
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/tables")
    parser.add_argument("--input", action="append", default=[], help="override name=path")
    parser.add_argument("--only", action="append", default=[], choices=TABLES)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    registry = ArtifactRegistry(args.root, ArtifactRegistry.parse_overrides(args.input))
    index = build_all(registry, args.output_dir, args.only or None)
    publish_text(args.output_dir / "table_index.json", json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"produced: {sorted(index['produced'])}")
    print("PENDING: " + (json.dumps(index["pending"], indent=2) if index["pending"] else "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit natural (no-injection) failures separately from injected results.

Reads derived extraction manifests (``data/derived/<dataset_id>/extraction_manifest.jsonl``)
for the development and validation datasets, lists every ``fault_family == none``
episode that carries a primary event, tabulates their classes, and, when alarm-applied
prediction tables are given, reports warning performance on exactly those episodes.
Results are never merged with injected results; the output says so explicitly.
Writes the immutable ``reports/confirmatory/natural_failure_audit.yaml``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_confirmatory import (
    episode_outcomes, lead_time_distribution, load_table, summarize, truth,
)
from src.dataset_inventory import publish_new_bytes

DEFAULT_DATASETS = (
    "balanced_pilot_v1-development-648",
    "balanced_validation_v1-validation-324",
)
TIMEOUT_CLASS = "mission_timeout"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def natural_failures(dataset_id: str, rows: list[dict]) -> tuple[list[dict], dict[str, object]]:
    if any(row.get("protected_test_used") is not False for row in rows):
        raise ValueError(f"{dataset_id}: manifest rows must declare protected_test_used: false")
    clean = [row for row in rows if row.get("fault_family") == "none"]
    failures = []
    for row in clean:
        event_class = row.get("primary_event_class")
        if not event_class:
            continue
        start = row.get("episode_start")
        event_time = row.get("primary_event_time")
        failures.append({
            "run_id": row["run_id"],
            "dataset_id": dataset_id,
            "split": row.get("split"),
            "map_id": row.get("map_id"),
            "route_id": row.get("route_id"),
            "seed": row.get("seed"),
            "system_id": row.get("system_id"),
            "dataset_episode_key": row.get("dataset_episode_key"),
            "primary_event_class": event_class,
            "primary_event_time": event_time,
            "episode_start": start,
            "episode_end": row.get("episode_end"),
            "seconds_from_episode_start_to_event": (
                float(event_time) - float(start) if event_time is not None and start is not None else None
            ),
            "positive_sequences": row.get("positive_sequences"),
            "is_timeout": event_class == TIMEOUT_CLASS,
        })
    summary = {
        "episodes_in_manifest": len(rows),
        "clean_episodes": len(clean),
        "natural_failures": len(failures),
        "natural_failure_rate_per_clean_episode": len(failures) / len(clean) if clean else None,
        "by_class": dict(sorted(Counter(item["primary_event_class"] for item in failures).items())),
        "by_map": dict(sorted(Counter(item["map_id"] for item in failures).items())),
        "by_system": dict(sorted(Counter(str(item["system_id"]) for item in failures).items())),
        "timeouts": sum(1 for item in failures if item["is_timeout"]),
    }
    return failures, summary


def warning_performance(path: Path, failures: list[dict]) -> dict[str, object]:
    rows = load_table(path)
    if any(truth(row["protected_test_used"]) for row in rows):
        raise ValueError(f"{path}: natural-failure audit accepts pre-protected tables only")
    wanted = {item["run_id"] for item in failures}
    subset = [row for row in rows if row["run_id"] in wanted]
    model_ids = sorted({row["model_id"] for row in rows})
    covered = {row["run_id"] for row in subset}
    result: dict[str, object] = {
        "prediction_table": str(path),
        "prediction_table_sha256": sha256_file(path),
        "model_id": model_ids[0] if len(model_ids) == 1 else model_ids,
        "natural_failures_in_table": len(covered),
        "natural_failures_missing_from_table": sorted(wanted - covered),
        "merged_with_injected_results": False,
    }
    if not subset:
        result["status"] = "no_natural_failure_episodes_in_table"
        return result
    outcomes = episode_outcomes(subset)
    non_timeout = [row for row in outcomes.values() if not row["is_timeout"]]
    timeouts = [row for row in outcomes.values() if row["is_timeout"]]
    by_class: dict[str, object] = {}
    for event_class in sorted({row["primary_event_class"] for row in outcomes.values() if row["has_event"]}):
        events = [row for row in outcomes.values() if row["primary_event_class"] == event_class]
        by_class[event_class] = {
            "event_count": len(events),
            "detected": sum(1 for row in events if row["detected"]),
            "recall": sum(1 for row in events if row["detected"]) / len(events),
            "lead_time": lead_time_distribution(events),
        }
    result.update({
        "status": "evaluated",
        "natural_failures_excluding_timeouts": summarize(non_timeout),
        "timeouts_analysed_separately": {
            "count": len(timeouts),
            "recall": (sum(1 for row in timeouts if row["detected"]) / len(timeouts)) if timeouts else None,
            "lead_time": lead_time_distribution(timeouts),
        },
        "by_class": by_class,
    })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", type=Path, default=ROOT / "data/derived")
    parser.add_argument("--dataset-id", action="append", dest="dataset_ids")
    parser.add_argument("--predictions", type=Path, action="append", default=[],
                        help="alarm-applied, pre-protected prediction table (repeatable)")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/confirmatory/natural_failure_audit.yaml")
    args = parser.parse_args(argv)
    dataset_ids = args.dataset_ids or list(DEFAULT_DATASETS)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable report {args.output}")

    failures: list[dict] = []
    datasets: dict[str, object] = {}
    inputs: dict[str, str] = {}
    for dataset_id in dataset_ids:
        manifest = args.derived_root / dataset_id / "extraction_manifest.jsonl"
        if not manifest.is_file():
            raise SystemExit(f"extraction manifest missing: {manifest}")
        rows = read_manifest(manifest)
        try:
            found, summary = natural_failures(dataset_id, rows)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        failures.extend(found)
        datasets[dataset_id] = {"manifest": str(manifest), **summary}
        inputs[str(manifest)] = sha256_file(manifest)
    performance = []
    for path in args.predictions:
        try:
            performance.append(warning_performance(path, failures))
        except ValueError as error:
            raise SystemExit(str(error)) from error
    report = {
        "schema_version": 1,
        "kind": "natural_failure_audit",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protected_test_used": False,
        "merged_with_injected_results": False,
        "source_condition": "no_injection",
        "analysis_unit": "episode",
        "label": "exploratory_ecological_validity",
        "inputs_sha256": inputs,
        "datasets": datasets,
        "totals": {
            "natural_failures": len(failures),
            "by_class": dict(sorted(Counter(item["primary_event_class"] for item in failures).items())),
            "by_split": dict(sorted(Counter(str(item["split"]) for item in failures).items())),
            "timeouts": sum(1 for item in failures if item["is_timeout"]),
        },
        "natural_failures": sorted(failures, key=lambda item: (item["dataset_id"], item["run_id"])),
        "warning_performance": performance,
    }
    publish_new_bytes(args.output, yaml.safe_dump(report, sort_keys=False).encode("utf-8"))
    print(f"wrote {args.output}: {len(failures)} natural failures across {len(datasets)} datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

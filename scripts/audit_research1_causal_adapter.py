#!/usr/bin/env python3
"""Audit every Research 1 development catalog row for causal-adapter admissibility.

Read-only on Research 1. Streams only the adapter topics of development bags listed in
``data/manifests/research1_temporal_catalog_v2.jsonl``; validation rows are counted
and never opened; protected ``test_*`` rows are absent from the catalog and refused
if encountered. Publishes an immutable report and the admitted-row manifest.

    source scripts/env_research2.sh
    nice -n 19 python3 scripts/audit_research1_causal_adapter.py [--workers 3] [--limit N]
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402
from src.research1_adapter import (  # noqa: E402
    ADAPTER_VERSION, DURATION_SIM_TIME_COMMIT, AdapterThresholds, EPISODE_START_DEFINITION,
    EVENT_TIME_DERIVATION, RESEARCH1_ROOT, TERMINAL_EVENT_CLASS, TIME_BASE, assess_episode,
    assessment_to_manifest_row, index_aggregates, load_aggregate, load_route_spec,
    read_bag_evidence, sha256_path,
)


CATALOG = ROOT / "data/manifests/research1_temporal_catalog_v2.jsonl"
REPORT = ROOT / "reports/integrity/research1_causal_adapter_audit_v1.yaml"
MANIFEST = ROOT / "data/manifests/research1_development_adapter_v1.jsonl"
EVENTS_CONFIG = ROOT / "configs/failure_events.yaml"


def duration_time_base_for_commit(commit_sha: str, cache: dict[str, str | None]) -> str | None:
    """Classify a Research 1 commit by read-only git ancestry; None when unknown."""
    commit = str(commit_sha or "").strip()
    if commit in cache:
        return cache[commit]
    base: str | None = None
    if commit:
        exists = subprocess.run(
            ["git", "-C", str(RESEARCH1_ROOT), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False, capture_output=True,
        ).returncode == 0
        if exists:
            descendant = subprocess.run(
                ["git", "-C", str(RESEARCH1_ROOT), "merge-base", "--is-ancestor",
                 DURATION_SIM_TIME_COMMIT, commit],
                check=False, capture_output=True,
            ).returncode == 0
            base = "simulation" if descendant else "wall_clock"
    cache[commit] = base
    return base


def research1_campaign_name(aggregate_path: Path) -> str:
    parent = aggregate_path.parent
    return "pilot_top_level" if parent == RESEARCH1_ROOT / "results/raw" else parent.name


def audit_row(row: dict, aggregate_path: str, duplicate: bool, niceness: int,
              duration_time_base: str | None) -> dict:
    if niceness:
        try:
            os.nice(niceness)
        except OSError:
            pass
    if not str(row["map_id"]).startswith("dev_") or row.get("split") != "development":
        raise ValueError(f"refusing non-development row {row['run_id']}")
    event_config = yaml.safe_load(EVENTS_CONFIG.read_text(encoding="utf-8"))
    aggregate = load_aggregate(Path(aggregate_path), str(row["run_id"]), str(row["map_id"]))
    route = load_route_spec(RESEARCH1_ROOT, str(row["map_id"]), str(row["route_id"]))
    bag_dir = RESEARCH1_ROOT / str(row["bag_path"])
    metadata_path = bag_dir / "metadata.yaml"
    if sha256_path(metadata_path) != row.get("metadata_sha256"):
        return {"run_id": row["run_id"], "admissible": False,
                "reasons": ["bag_metadata_checksum_changed"], "mapped_event_class": None}
    started = time.monotonic()
    evidence = read_bag_evidence(bag_dir)
    assessment = assess_episode(
        row, aggregate, route, evidence, event_config=event_config,
        thresholds=AdapterThresholds(), duplicate_run_id=duplicate,
        duration_time_base=duration_time_base,
        research1_campaign=research1_campaign_name(Path(aggregate_path)),
    )
    assessment["aggregate_sha256"] = sha256_path(Path(aggregate_path))
    assessment["bag_read_seconds"] = round(time.monotonic() - started, 3)
    return assessment


def _stats(values: list[float]) -> dict | None:
    finite = [float(v) for v in values if v is not None]
    if not finite:
        return None
    return {
        "count": len(finite), "minimum": min(finite), "median": statistics.median(finite),
        "maximum": max(finite),
        "p05": statistics.quantiles(finite, n=20)[0] if len(finite) >= 20 else min(finite),
        "p95": statistics.quantiles(finite, n=20)[-1] if len(finite) >= 20 else max(finite),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--niceness", type=int, default=19)
    parser.add_argument("--limit", type=int, default=None,
                        help="audit only the first N development rows (engineering smoke)")
    args = parser.parse_args()
    if args.workers > 3:
        raise SystemExit("at most 3 parallel bag readers are permitted while the campaign runs")
    if args.report.exists() or args.manifest.exists():
        raise SystemExit(f"refusing to overwrite {args.report} or {args.manifest}")

    rows = [json.loads(line) for line in args.catalog.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if any(str(row.get("map_id", "")).startswith("test_") for row in rows):
        raise SystemExit("catalog contains protected rows; refusing")
    development = [row for row in rows if row.get("split") == "development"]
    validation_count = sum(row.get("split") == "validation" for row in rows)
    if args.limit is not None:
        development = development[: args.limit]
    duplicates = {key for key, count in Counter(r["run_id"] for r in development).items() if count > 1}
    aggregates = index_aggregates(RESEARCH1_ROOT)
    missing_aggregate = [row["run_id"] for row in development if row["run_id"] not in aggregates]

    started_wall = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    commit_cache: dict[str, str | None] = {}
    time_bases: dict[str, str | None] = {}
    for row in development:
        path = aggregates.get(row["run_id"])
        if path is None:
            continue
        aggregate = load_aggregate(path, str(row["run_id"]), str(row["map_id"]))
        time_bases[row["run_id"]] = duration_time_base_for_commit(
            str(aggregate.get("commit_sha", "")), commit_cache
        )
    results: dict[str, dict] = {}
    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {}
        for row in development:
            if row["run_id"] in aggregates:
                futures[pool.submit(
                    audit_row, row, str(aggregates[row["run_id"]]),
                    row["run_id"] in duplicates, args.niceness, time_bases[row["run_id"]],
                )] = row["run_id"]
        done = 0
        for future in as_completed(futures):
            run_id = futures[future]
            done += 1
            try:
                results[run_id] = future.result()
            except Exception as error:  # noqa: BLE001 - recorded, never hidden
                failures[run_id] = f"{type(error).__name__}: {error}"
            if done % 50 == 0 or done == len(futures):
                print(f"[{done}/{len(futures)}] audited", flush=True)
    for run_id in missing_aggregate:
        results[run_id] = {"run_id": run_id, "admissible": False,
                           "reasons": ["aggregate_missing"], "mapped_event_class": None}
    for run_id, message in failures.items():
        results[run_id] = {"run_id": run_id, "admissible": False,
                           "reasons": [f"audit_error:{message[:120]}"], "mapped_event_class": None}
    wall_seconds = time.monotonic() - started_wall

    by_row = {row["run_id"]: row for row in development}
    admitted_rows = []
    reason_counts: Counter[str] = Counter()
    first_reason_counts: Counter[str] = Counter()
    strata: dict[str, dict[str, dict[str, int]]] = {
        key: {} for key in ("map_id", "route_id", "system_id", "shift_family",
                            "research1_campaign", "aggregate_duration_time_base",
                            "research1_commit_sha")
    }
    mapped_all: Counter[str] = Counter()
    mapped_admitted: Counter[str] = Counter()
    primary_admitted: Counter[str] = Counter()
    exactness_admitted: Counter[str] = Counter()
    start_method_admitted: Counter[str] = Counter()
    missing_channel_counts: Counter[str] = Counter()
    rtf, collision_residuals, duration_residuals, start_lags, plan_lags, read_seconds = (
        [], [], [], [], [], []
    )
    for run_id in sorted(results):
        assessment = results[run_id]
        row = by_row[run_id]
        admissible = bool(assessment.get("admissible"))
        reasons = list(assessment.get("reasons", []))
        for reason in reasons:
            reason_counts[reason] += 1
        if reasons:
            first_reason_counts[reasons[0]] += 1
        mapped = assessment.get("mapped_event_class")
        mapped_all[str(mapped)] += 1
        for key in strata:
            value = str(assessment.get(key, row.get(key, "")) or row.get(key, ""))
            bucket = strata[key].setdefault(value, {"rows": 0, "admitted": 0})
            bucket["rows"] += 1
            bucket["admitted"] += admissible
        if assessment.get("real_time_factor") is not None:
            rtf.append(assessment["real_time_factor"])
        if assessment.get("bag_read_seconds") is not None:
            read_seconds.append(assessment["bag_read_seconds"])
        if admissible:
            mapped_admitted[str(mapped)] += 1
            primary_admitted[str(assessment.get("primary_event_class"))] += 1
            exactness_admitted[str(assessment.get("terminal_time_exactness"))] += 1
            start_method_admitted[str(assessment.get("episode_start_method"))] += 1
            for channel in assessment.get("missing_channels", []):
                missing_channel_counts[channel] += 1
            collision_residuals.append(assessment.get("collision_residual_s"))
            duration_residuals.append(assessment.get("duration_residual_s"))
            start_lags.append(assessment.get("start_lag_after_bag_start_s"))
            plan_lags.append(assessment.get("first_plan_minus_start_s"))
            admitted_rows.append(assessment_to_manifest_row(
                row, assessment, aggregate_path=aggregates[run_id],
                aggregate_sha256=assessment["aggregate_sha256"],
            ))
    admitted_rows.sort(key=lambda item: (item["map_id"], item["route_id"], item["run_id"]))
    manifest_bytes = "".join(
        json.dumps(item, sort_keys=True, separators=(",", ":"), default=float) + "\n"
        for item in admitted_rows
    ).encode("utf-8")

    thresholds = AdapterThresholds()
    report = {
        "schema_version": 1,
        "audit_type": "research1_causal_adapter_admissibility",
        "adapter_version": ADAPTER_VERSION,
        "status": "complete" if args.limit is None and not failures else "partial",
        "protected_outcomes_consulted": False,
        "protected_test_used": False,
        "validation_rows_counted_only_never_opened": validation_count,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": round(wall_seconds, 1),
        "workers": args.workers,
        "niceness": args.niceness,
        "research1_root": str(RESEARCH1_ROOT),
        "sources": {
            "catalog": str(args.catalog.relative_to(ROOT)) if args.catalog.is_relative_to(ROOT) else str(args.catalog),
            "catalog_sha256": sha256_file(args.catalog),
            "failure_events_sha256": sha256_file(EVENTS_CONFIG),
            "feature_schema_sha256": sha256_file(ROOT / "configs/feature_schema.yaml"),
            "route_specs": {
                map_id: sha256_file(RESEARCH1_ROOT / "configs/routes" / f"{map_id}.yaml")
                for map_id in sorted({row["map_id"] for row in development})
                if (RESEARCH1_ROOT / "configs/routes" / f"{map_id}.yaml").is_file()
            },
        },
        "counts": {
            "catalog_rows": len(rows),
            "development_rows": len(development),
            "development_rows_audited": len(results),
            "admitted": len(admitted_rows),
            "rejected": len(results) - len(admitted_rows),
            "audit_errors": len(failures),
            "duplicate_run_ids": len(duplicates),
            "missing_aggregates": len(missing_aggregate),
        },
        "rejection_reasons_all_occurrences": dict(sorted(reason_counts.items())),
        "rejection_reasons_first_per_row": dict(sorted(first_reason_counts.items())),
        "by_map": dict(sorted(strata["map_id"].items())),
        "by_route": dict(sorted(strata["route_id"].items())),
        "by_system": dict(sorted(strata["system_id"].items())),
        "by_shift_family": dict(sorted(strata["shift_family"].items())),
        "by_research1_campaign_directory": dict(sorted(strata["research1_campaign"].items())),
        "by_aggregate_duration_time_base": dict(sorted(strata["aggregate_duration_time_base"].items())),
        "by_research1_commit": dict(sorted(strata["research1_commit_sha"].items())),
        "duration_time_base_rule": (
            f"duration_s is simulated seconds for descendants of Research 1 commit "
            f"{DURATION_SIM_TIME_COMMIT[:12]} (protocol v1.21, 2026-08-26) and wall seconds "
            "before it; wall durations are converted through the per-episode clock map"
        ),
        "mapped_event_class_all_rows": dict(sorted(mapped_all.items())),
        "mapped_event_class_admitted": dict(sorted(mapped_admitted.items())),
        "primary_event_class_admitted_after_operational_precedence": dict(sorted(primary_admitted.items())),
        "terminal_time_exactness_admitted": dict(sorted(exactness_admitted.items())),
        "episode_start_method_admitted": dict(sorted(start_method_admitted.items())),
        "missing_channels_admitted": dict(sorted(missing_channel_counts.items())),
        "time_base": TIME_BASE,
        "episode_start_definition": EPISODE_START_DEFINITION,
        "event_time_derivation": EVENT_TIME_DERIVATION,
        "terminal_state_mapping": TERMINAL_EVENT_CLASS,
        "fault_family_rule": "none when shift_family == clean, otherwise research1_<shift_family>; never a Research 2 family",
        "thresholds": thresholds.__dict__,
        "distributions": {
            "real_time_factor_sim_per_wall": _stats(rtf),
            "collision_contact_minus_aggregate_terminal_s": _stats(collision_residuals),
            "aggregate_terminal_minus_last_command_s": _stats(duration_residuals),
            "episode_start_after_bag_start_s": _stats(start_lags),
            "first_plan_minus_episode_start_s": _stats(plan_lags),
            "bag_read_seconds": _stats(read_seconds),
        },
        "admitted_manifest": {
            "path": str(args.manifest.relative_to(ROOT)) if args.manifest.is_relative_to(ROOT) else str(args.manifest),
            "sha256": __import__("hashlib").sha256(manifest_bytes).hexdigest(),
            "rows": len(admitted_rows),
        },
        "admission_status": "causal_adapter_admitted_pending_human_gate",
        "failures": failures,
        "interpretation": (
            "Mechanical admissibility only. Admitted rows are Research 1 development "
            "episodes expressible on the Research 2 causal grid; they are natural/shift "
            "episodes, not Research 2 fault injections, and enter fitting only after the "
            "human gate. Retained Research 1 bags are outcome-selected (mostly failures), "
            "so their event prevalence must be reported separately."
        ),
    }
    publish_new_bytes(args.manifest, manifest_bytes)
    publish_new_bytes(args.report, yaml.safe_dump(report, sort_keys=False).encode("utf-8"))
    print(
        f"RESEARCH 1 CAUSAL ADAPTER AUDIT: {len(admitted_rows)}/{len(results)} development "
        f"rows admitted; {len(failures)} audit errors; {wall_seconds:.0f} s -> {args.report}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

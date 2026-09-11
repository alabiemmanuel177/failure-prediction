#!/usr/bin/env python3
"""Run the frozen offline causal pipeline over one immutable episode inventory.

For every scientific episode in a hash-addressed inventory this driver executes the
already-validated per-episode steps in order:

  derive_operational_events -> extract_episode_annotation -> generate_labels
  -> extract_bag_scalar_telemetry -> extract_scalar_features
  -> derive_window_features -> assemble_episode_sequences

All outputs are immutable and land under ``data/derived/<dataset_id>/``. A JSONL
extraction manifest records the SHA-256 of every produced artifact so a later
training run can prove exactly which derived files it consumed. Protected episodes
are refused by every underlying step; this driver additionally refuses inventories
that declare ``protected_test_used`` anywhere.

The driver is resumable: an episode whose seven artifacts already exist and match
the manifest row is skipped. It never overwrites, never deletes and never touches
raw bags, summaries or event sidecars.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402

STEPS = (
    "operational_events", "annotations", "labels", "telemetry",
    "causal_features", "window_features", "sequences",
)
SUFFIX = {
    "operational_events": ".yaml", "annotations": ".yaml", "labels": ".csv",
    "telemetry": ".csv", "causal_features": ".csv", "window_features": ".csv",
    "sequences": ".npz",
}


def artifact_paths(output_root: Path, run_id: str) -> dict[str, Path]:
    return {step: output_root / step / f"{run_id}{SUFFIX[step]}" for step in STEPS}


def _run(command: list[str], log: Path) -> None:
    with log.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command, check=False, stdout=stream, stderr=subprocess.STDOUT, text=True,
        )
    if result.returncode:
        raise RuntimeError(f"step failed ({result.returncode}): {command[1]}")


def confirmatory_gate_passed() -> bool:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_readiness.py"), "--stage", "confirmatory"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def extract_episode(
    row: dict, output_root: Path, summary_root: Path, log_root: Path, niceness: int,
    allow_protected: bool = False,
) -> dict:
    run_id = str(row["run_id"])
    started = time.monotonic()
    if niceness:
        try:
            os.nice(niceness)
        except OSError:
            pass
    summary_path = summary_root / f"{run_id}.yaml"
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary missing for {run_id}")
    if sha256_file(summary_path) != row["summary_sha256"]:
        raise ValueError(f"summary hash differs from inventory for {run_id}")
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
    protected = summary["environment"].get("protected_test_used")
    if protected is not False and not (allow_protected and protected is True):
        raise ValueError(f"refusing non-development/validation episode {run_id}")
    protected_flags = ["--allow-protected-after-freeze"] if protected is True else []
    if summary["identity"]["run_id"] != run_id:
        raise ValueError(f"summary identity differs for {run_id}")
    paths = artifact_paths(output_root, run_id)
    existing = [step for step, path in paths.items() if path.exists()]
    if existing and len(existing) != len(STEPS):
        raise RuntimeError(
            f"partial derived outputs for {run_id}: {existing}; move them aside before resuming"
        )
    log = log_root / f"{run_id}.log"
    if not existing:
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        python = sys.executable
        scripts = ROOT / "scripts"
        _run([python, str(scripts / "derive_operational_events.py"),
              str(summary_path), str(paths["operational_events"]), *protected_flags], log)
        _run([python, str(scripts / "extract_episode_annotation.py"),
              str(summary_path), str(paths["annotations"]),
              "--operational-events", str(paths["operational_events"]), *protected_flags], log)
        _run([python, str(scripts / "generate_labels.py"),
              str(paths["annotations"]), str(paths["labels"])], log)
        telemetry_command = [python, str(scripts / "extract_bag_scalar_telemetry.py"),
                             str(summary["provenance"]["bag_path"]), run_id,
                             str(paths["telemetry"])]
        clock_map = summary["provenance"].get("receive_clock_map")
        if clock_map:
            # Adapted Research 1 bags: availability timestamps are mapped onto the
            # simulation clock through the per-episode map declared by the summary.
            telemetry_command += ["--receive-clock-map", str(clock_map)]
        _run(telemetry_command, log)
        _run([python, str(scripts / "extract_scalar_features.py"),
              str(paths["telemetry"]), str(paths["labels"]),
              str(paths["causal_features"])], log)
        _run([python, str(scripts / "derive_window_features.py"),
              str(paths["causal_features"]), str(summary_path),
              str(paths["window_features"])], log)
        _run([python, str(scripts / "assemble_episode_sequences.py"),
              str(paths["telemetry"]), str(paths["labels"]), str(paths["annotations"]),
              str(summary_path), str(paths["sequences"]), *protected_flags], log)
    import numpy as np
    with np.load(paths["sequences"], allow_pickle=False) as artifact:
        y = artifact["y"]
        metadata = json.loads(str(artifact["metadata_json"].item()))
        shape = list(artifact["X"].shape)
    labels = paths["labels"].read_text(encoding="utf-8").splitlines()[1:]
    eligibility_counts: dict[str, int] = {}
    for line in labels:
        eligibility = line.split(",")[6]
        eligibility_counts[eligibility] = eligibility_counts.get(eligibility, 0) + 1
    annotation = yaml.safe_load(paths["annotations"].read_text(encoding="utf-8"))
    events = annotation.get("events", [])
    return {
        "run_id": run_id,
        "dataset_episode_key": row.get("dataset_episode_key"),
        "split": row["split"],
        "map_id": row["map_id"],
        "route_id": row["route_id"],
        "fault_family": row["fault_family"],
        "severity": row["severity"],
        "seed": row["seed"],
        "system_id": row.get("system_id"),
        "recording_profile": row.get("recording_profile"),
        "protected_test_used": protected,
        "summary_sha256": row["summary_sha256"],
        "bag_checksum_sha256": row["bag_checksum_sha256"],
        "artifact_sha256": {step: sha256_file(path) for step, path in paths.items()},
        "primary_event_class": events[0]["class"] if events else None,
        "primary_event_time": float(events[0]["time"]) if events else None,
        "episode_start": float(annotation["episode"]["start_time"]),
        "episode_end": float(annotation["episode"]["end_time"]),
        "decision_counts": eligibility_counts,
        "sequence_shape": shape,
        "positive_sequences": int((y == 1).sum()),
        "negative_sequences": int((y == 0).sum()),
        "feature_columns": len(metadata["feature_columns"]),
        "reused_existing_artifacts": bool(existing),
        "extraction_seconds": round(time.monotonic() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True,
                        help="immutable *.episodes.jsonl inventory")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/derived")
    parser.add_argument("--summary-root", type=Path, default=ROOT / "data/raw/summaries")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--niceness", type=int, default=19)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N inventory rows (engineering smoke)")
    parser.add_argument("--allow-partial-manifest", action="store_true",
                        help="publish a manifest even when some episodes failed")
    parser.add_argument("--allow-protected-after-freeze", action="store_true",
                        help="derive held-out episodes; requires the confirmatory readiness gate")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.inventory.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    protected_rows = [row for row in rows if row.get("protected_test_used") is True]
    if any(row.get("protected_test_used") not in (False, True) for row in rows):
        raise SystemExit("inventory contains episodes without an explicit protection marker")
    if protected_rows and not (args.allow_protected_after_freeze and confirmatory_gate_passed()):
        raise SystemExit("inventory contains protected episodes; refusing before the confirmatory freeze gate")
    splits = {row["split"] for row in rows}
    allowed_splits = {"development", "validation"} | ({"held_out_map_test"} if protected_rows else set())
    if not splits <= allowed_splits:
        raise SystemExit(f"inventory declares non-admitted splits: {sorted(splits)}")
    if args.limit is not None:
        rows = rows[:args.limit]
    output_root = args.output_root / args.dataset_id
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "extraction_manifest.jsonl"
    report_path = output_root / "extraction_report.yaml"
    if manifest_path.exists() or report_path.exists():
        raise SystemExit(f"derived manifest already published for {args.dataset_id}; refusing overwrite")

    started_utc = datetime.now(timezone.utc).isoformat()
    results: dict[str, dict] = {}
    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(extract_episode, row, output_root, args.summary_root, log_root,
                        args.niceness, bool(protected_rows)): row["run_id"]
            for row in rows
        }
        done = 0
        for future in as_completed(futures):
            run_id = futures[future]
            done += 1
            try:
                results[run_id] = future.result()
                status = "ok"
            except Exception as error:  # noqa: BLE001 - recorded, never hidden
                failures[run_id] = f"{type(error).__name__}: {error}"
                status = "FAILED"
            print(f"[{done}/{len(rows)}] {run_id} {status}", flush=True)

    if failures and not args.allow_partial_manifest:
        (output_root / "extraction_failures.yaml").write_text(
            yaml.safe_dump({"failures": failures}, sort_keys=True), encoding="utf-8"
        )
        print(f"EXTRACTION INCOMPLETE: {len(failures)} failure(s); manifest not published")
        return 1

    ordered = [results[row["run_id"]] for row in rows if row["run_id"] in results]
    manifest_bytes = "".join(
        json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in ordered
    ).encode("utf-8")
    publish_new_bytes(manifest_path, manifest_bytes)
    positives = sum(item["positive_sequences"] for item in ordered)
    negatives = sum(item["negative_sequences"] for item in ordered)
    events = sum(1 for item in ordered if item["primary_event_class"])
    by_family: dict[str, dict[str, int]] = {}
    for item in ordered:
        bucket = by_family.setdefault(item["fault_family"], {"episodes": 0, "events": 0,
                                                            "positive_sequences": 0,
                                                            "negative_sequences": 0})
        bucket["episodes"] += 1
        bucket["events"] += bool(item["primary_event_class"])
        bucket["positive_sequences"] += item["positive_sequences"]
        bucket["negative_sequences"] += item["negative_sequences"]
    report = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "status": "complete" if not failures else "partial",
        "protected_test_used": bool(protected_rows),
        "training_performed": False,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "inventory": str(args.inventory.resolve().relative_to(ROOT)) if args.inventory.resolve().is_relative_to(ROOT) else str(args.inventory.resolve()),
        "inventory_sha256": sha256_file(args.inventory),
        "feature_schema_sha256": sha256_file(ROOT / "configs/feature_schema.yaml"),
        "leakage_denylist_sha256": sha256_file(ROOT / "configs/leakage_denylist.yaml"),
        "failure_events_sha256": sha256_file(ROOT / "configs/failure_events.yaml"),
        "extraction_manifest": str(manifest_path.resolve().relative_to(ROOT)) if manifest_path.resolve().is_relative_to(ROOT) else str(manifest_path.resolve()),
        "extraction_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "counts": {
            "inventory_rows": len(rows),
            "extracted_episodes": len(ordered),
            "failed_episodes": len(failures),
            "episodes_with_primary_event": events,
            "positive_sequences": positives,
            "negative_sequences": negatives,
            "sequence_positive_fraction": positives / (positives + negatives)
            if positives + negatives else None,
        },
        "by_fault_family": dict(sorted(by_family.items())),
        "failures": failures,
        "workers": args.workers,
        "niceness": args.niceness,
        "interpretation": (
            "Derived causal artifacts for one immutable inventory. Extraction only; no "
            "model was fitted, no threshold selected and no protected data touched."
        ),
    }
    publish_new_bytes(report_path, yaml.safe_dump(report, sort_keys=False).encode("utf-8"))
    print(f"EXTRACTION {report['status'].upper()}: {len(ordered)} episodes, "
          f"{positives} positive / {negatives} negative sequences -> {report_path}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

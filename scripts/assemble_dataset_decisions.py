#!/usr/bin/env python3
"""Publish all-decisions causal sequence artifacts for one derived dataset.

`sequences/<run_id>.npz` holds only eligible (label 0/1) decisions and is the fitting
set. Deployed inference and the alarm policy, however, run over every consecutive
2 Hz decision. This driver therefore builds `decisions/<run_id>.npz` for every
label-grid decision of every episode listed in `extraction_manifest.jsonl`, using the
identical causal history construction (`assemble_causal_sequences` with
`include_ineligible=True`), and proves for each episode that the eligible rows are
numerically identical to the fitting artifact. Excluded decisions carry label -1 and
their eligibility string so that metrics can ignore them while the policy still sees
them. Outputs are immutable; nothing raw is touched.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402
from src.features import (  # noqa: E402
    LeakagePolicy, ScalarSample, assemble_causal_sequences, load_primary_feature_set,
    load_raw_feature_contract, model_columns, resolve_feature_specs,
    validate_decision_artifact,
)
from src.protected_data import enforce_protected_boundary  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from assemble_episode_sequences import publish_npz  # noqa: E402


def build_episode(dataset_root: Path, row: dict, summary_root: Path,
                  windowing: dict, primary: tuple[str, ...], policy: LeakagePolicy,
                  contract: dict, schema_sha: str, denylist_sha: str,
                  allow_protected: bool = False, gate_passed: bool = False) -> dict:
    run_id = row["run_id"]
    telemetry_path = dataset_root / "telemetry" / f"{run_id}.csv"
    labels_path = dataset_root / "labels" / f"{run_id}.csv"
    annotation_path = dataset_root / "annotations" / f"{run_id}.yaml"
    summary_path = summary_root / f"{run_id}.yaml"
    output = dataset_root / "decisions" / f"{run_id}.npz"
    for name, path in (("telemetry", telemetry_path), ("labels", labels_path),
                       ("annotations", annotation_path)):
        if sha256_file(path) != row["artifact_sha256"][name]:
            raise ValueError(f"{run_id}: {name} artifact differs from extraction manifest")
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
    protected = summary["environment"].get("protected_test_used")
    enforce_protected_boundary(protected, explicitly_allowed=allow_protected,
                               confirmatory_gate_passed=gate_passed)
    annotation = yaml.safe_load(annotation_path.read_text(encoding="utf-8"))
    telemetry = list(csv.DictReader(telemetry_path.open(newline="", encoding="utf-8")))
    labels = list(csv.DictReader(labels_path.open(newline="", encoding="utf-8")))
    specs, grouped = resolve_feature_specs(contract, telemetry)
    samples = {
        name: [ScalarSample(float(item["timestamp"]), float(item["value"])) for item in rows]
        for name, rows in grouped.items()
    }
    eligibility_by_index: dict[int, str] = {}
    for item in labels:
        raw_label = item.get("label")
        item["label"] = int(raw_label) if raw_label in {"0", "1"} else None
        item["decision_index"] = int(item["decision_index"])
        item["decision_time"] = float(item["decision_time"])
        eligibility_by_index[item["decision_index"]] = item["eligibility"]
    goal = summary["environment"]["goal_pose"]
    examples = assemble_causal_sequences(
        run_id=run_id,
        episode_start=float(annotation["episode"]["start_time"]),
        episode_end=float(annotation["episode"]["end_time"]),
        labels=labels, specs=specs, samples_by_feature=samples, leakage_policy=policy,
        primary_features=primary, goal_x=float(goal["x"]), goal_y=float(goal["y"]),
        history_seconds=float(windowing["history_seconds"]),
        stride_seconds=float(windowing["decision_stride_seconds"]),
        include_ineligible=True,
    )
    columns = model_columns(primary)
    time_steps = round(float(windowing["history_seconds"]) / float(windowing["decision_stride_seconds"]))
    x = (np.asarray([example.values for example in examples], dtype=np.float32)
         if examples else np.empty((0, time_steps, len(columns)), dtype=np.float32))
    y = np.asarray([example.label for example in examples], dtype=np.int8)
    decision_index = np.asarray([example.decision_index for example in examples], dtype=np.int64)
    eligibility = np.asarray([eligibility_by_index[int(index)] for index in decision_index])

    # Prove equality with the fitting artifact on the eligible subset.
    with np.load(dataset_root / "sequences" / f"{run_id}.npz", allow_pickle=False) as fitting:
        eligible = y >= 0
        if not np.array_equal(fitting["decision_index"], decision_index[eligible]):
            raise AssertionError(f"{run_id}: eligible decision identities differ from sequences")
        if not np.array_equal(fitting["y"], y[eligible]):
            raise AssertionError(f"{run_id}: eligible labels differ from sequences")
        if not np.array_equal(fitting["X"], x[eligible]):
            raise AssertionError(f"{run_id}: eligible histories differ from sequences")

    events = annotation.get("events", [])
    metadata = {
        "schema_version": 1,
        "artifact_kind": "all_decisions",
        "run_id": run_id,
        "split": summary["environment"].get("split"),
        "map_id": summary["environment"].get("map_id"),
        "route_id": summary["environment"].get("route_id"),
        "feature_schema_sha256": schema_sha,
        "leakage_denylist_sha256": denylist_sha,
        "telemetry_sha256": row["artifact_sha256"]["telemetry"],
        "labels_sha256": row["artifact_sha256"]["labels"],
        "annotation_sha256": row["artifact_sha256"]["annotations"],
        "summary_sha256": row["summary_sha256"],
        "sequences_sha256": row["artifact_sha256"]["sequences"],
        "history_seconds": float(windowing["history_seconds"]),
        "stride_seconds": float(windowing["decision_stride_seconds"]),
        "sequence_count": int(len(examples)),
        "time_steps": time_steps,
        "feature_columns": list(columns),
        "protected_test_used": protected,
        "episode_start": float(annotation["episode"]["start_time"]),
        "episode_end": float(annotation["episode"]["end_time"]),
        "primary_event_class": events[0]["class"] if events else None,
        "primary_event_time": float(events[0]["time"]) if events else None,
    }
    publish_npz(output, {
        "X": x, "y": y, "eligibility": eligibility,
        "decision_index": decision_index,
        "decision_time": np.asarray([example.decision_time for example in examples], dtype=np.float64),
        "feature_names": np.asarray(columns),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    })
    findings = validate_decision_artifact(output, expected_feature_names=columns,
                                          allow_protected=protected is True)
    if findings:
        raise AssertionError(f"{run_id}: {findings}")
    counts: dict[str, int] = {}
    for name in eligibility.tolist():
        counts[name] = counts.get(name, 0) + 1
    return {
        "run_id": run_id,
        "decisions_sha256": sha256_file(output),
        "decision_count": int(len(examples)),
        "eligibility_counts": dict(sorted(counts.items())),
        "eligible_rows_identical_to_sequences": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--derived-root", type=Path, default=ROOT / "data/derived")
    parser.add_argument("--summary-root", type=Path, default=ROOT / "data/raw/summaries")
    parser.add_argument("--allow-protected-after-freeze", action="store_true")
    args = parser.parse_args()
    gate_passed = False
    if args.allow_protected_after_freeze:
        import subprocess
        gate_passed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_readiness.py"), "--stage", "confirmatory"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
    dataset_root = args.derived_root / args.dataset_id
    manifest_path = dataset_root / "extraction_manifest.jsonl"
    output_manifest = dataset_root / "decisions_manifest.jsonl"
    if output_manifest.exists():
        raise SystemExit(f"decisions manifest already published: {output_manifest}")
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    schema_path = ROOT / "configs/feature_schema.yaml"
    contract = load_raw_feature_contract(schema_path)
    primary = load_primary_feature_set(schema_path)
    policy = LeakagePolicy.from_yaml(ROOT / "configs/leakage_denylist.yaml")
    windowing = yaml.safe_load((ROOT / "configs/failure_events.yaml").read_text(encoding="utf-8"))["windowing"]
    schema_sha = sha256_file(schema_path)
    denylist_sha = sha256_file(ROOT / "configs/leakage_denylist.yaml")
    (dataset_root / "decisions").mkdir(parents=True, exist_ok=True)
    results = []
    for number, row in enumerate(rows, 1):
        results.append(build_episode(dataset_root, row, args.summary_root, windowing, primary,
                                     policy, contract, schema_sha, denylist_sha,
                                     args.allow_protected_after_freeze, gate_passed))
        if number % 50 == 0 or number == len(rows):
            print(f"[{number}/{len(rows)}] decisions assembled", flush=True)
    payload = "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                      for item in results).encode("utf-8")
    publish_new_bytes(output_manifest, payload)
    totals: dict[str, int] = {}
    for item in results:
        for name, count in item["eligibility_counts"].items():
            totals[name] = totals.get(name, 0) + count
    report = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "artifact_kind": "all_decisions",
        "status": "complete",
        "protected_test_used": any(r.get("protected_test_used") is True for r in rows),
        "training_performed": False,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "extraction_manifest_sha256": sha256_file(manifest_path),
        "decisions_manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "episodes": len(results),
        "decisions": sum(item["decision_count"] for item in results),
        "eligibility_totals": dict(sorted(totals.items())),
        "eligible_rows_identical_to_sequences_for_every_episode": True,
    }
    publish_new_bytes(dataset_root / "decisions_report.yaml",
                      yaml.safe_dump(report, sort_keys=False).encode("utf-8"))
    print(f"DECISIONS COMPLETE: {report['episodes']} episodes, {report['decisions']} decisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

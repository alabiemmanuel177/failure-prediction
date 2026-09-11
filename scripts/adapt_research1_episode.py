#!/usr/bin/env python3
"""Adapt one admitted Research 1 development bag into Research 2 episode artifacts.

For a run_id listed in ``data/manifests/research1_development_adapter_v1.jsonl`` this
writes, under ``<output-root>/`` (default ``data/adapted/research1_development_v1``):

  summaries/<run_id>.yaml      Research 2-style summary (identity/environment/label_only/
                               outcome/provenance, protected_test_used false)
  events/<run_id>.json         event sidecar with goal_dispatched and terminal_event
  topic_health/<run_id>.json   recorder topic message counts
  clock_maps/<run_id>.json     receive-time -> simulation-time map used for every
                               availability timestamp of this episode

The bag is re-read and re-assessed; the result must agree with the admitted manifest
row. The Research 1 bag is never modified or copied. Existing outputs are refused.

    source scripts/env_research2.sh
    nice -n 19 python3 scripts/adapt_research1_episode.py --run-id <uuid> [--output-root DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402
from src.research1_adapter import (  # noqa: E402
    ADAPTER_VERSION, AdapterThresholds, RESEARCH1_ROOT, ReceiveClockMap, assess_episode,
    build_event_sidecar, build_summary, load_aggregate, load_route_spec, read_bag_evidence,
    sha256_path,
)


MANIFEST = ROOT / "data/manifests/research1_development_adapter_v1.jsonl"
DEFAULT_OUTPUT_ROOT = ROOT / "data/adapted/research1_development_v1"
EVENTS_CONFIG = ROOT / "configs/failure_events.yaml"
COMPARED_FIELDS = (
    "terminal_state", "mapped_event_class", "terminal_event_class", "primary_event_class",
    "episode_start_method", "aggregate_duration_time_base", "fault_family", "severity",
)
COMPARED_TIMES = ("episode_start_sim", "episode_end_sim", "terminal_event_time_sim",
                  "primary_event_time_sim")


def adapter_config_hash() -> str:
    digest = hashlib.sha256()
    digest.update(f"research1_adapter_v{ADAPTER_VERSION}\n".encode("utf-8"))
    for name in ("configs/failure_events.yaml", "configs/feature_schema.yaml"):
        digest.update(sha256_file(ROOT / name).encode("utf-8"))
    return digest.hexdigest()


def adapt(row: dict, output_root: Path, thresholds: AdapterThresholds) -> dict:
    run_id = str(row["run_id"])
    if row.get("protected_test_used") is not False or row.get("split") != "development":
        raise ValueError(f"refusing non-development adapter row {run_id}")
    if not str(row["map_id"]).startswith("dev_"):
        raise ValueError(f"refusing protected or unknown map {row['map_id']}")
    outputs = {
        "summary": output_root / "summaries" / f"{run_id}.yaml",
        "events": output_root / "events" / f"{run_id}.json",
        "topic_health": output_root / "topic_health" / f"{run_id}.json",
        "clock_map": output_root / "clock_maps" / f"{run_id}.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite adapted artifacts: {existing}")
    bag_dir = RESEARCH1_ROOT / str(row["bag_path"])
    metadata_path = bag_dir / "metadata.yaml"
    if sha256_path(metadata_path) != row["metadata_sha256"]:
        raise ValueError(f"bag metadata checksum changed since the audit: {run_id}")
    aggregate_path = Path(row["aggregate_path"])
    if sha256_path(aggregate_path) != row["aggregate_sha256"]:
        raise ValueError(f"aggregate checksum changed since the audit: {run_id}")
    aggregate = load_aggregate(aggregate_path, run_id, str(row["map_id"]))
    route = load_route_spec(RESEARCH1_ROOT, str(row["map_id"]), str(row["route_id"]))
    if route is None:
        raise ValueError(f"route specification missing for {run_id}")
    event_config = yaml.safe_load(EVENTS_CONFIG.read_text(encoding="utf-8"))
    evidence = read_bag_evidence(bag_dir)
    assessment = assess_episode(
        row, aggregate, route, evidence, event_config=event_config, thresholds=thresholds,
        duration_time_base=row.get("aggregate_duration_time_base"),
        research1_campaign=row.get("research1_campaign"),
    )
    if not assessment["admissible"]:
        raise ValueError(f"{run_id} is no longer admissible: {assessment['reasons']}")
    for field in COMPARED_FIELDS:
        if assessment.get(field) != row.get(field):
            raise ValueError(f"{run_id}: {field} differs from the admitted manifest")
    for field in COMPARED_TIMES:
        left, right = assessment.get(field), row.get(field)
        if (left is None) != (right is None) or (
            left is not None and abs(float(left) - float(right)) > 1e-6
        ):
            raise ValueError(f"{run_id}: {field} differs from the admitted manifest")
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))["rosbag2_bagfile_information"]
    mcap_files = [bag_dir / name for name in metadata["relative_file_paths"]]
    if len(mcap_files) != 1 or not mcap_files[0].is_file():
        raise ValueError(f"{run_id}: expected exactly one MCAP file")
    bag_checksum = sha256_path(mcap_files[0])

    clock_payload = json.dumps(
        ReceiveClockMap(evidence.clock_knots).to_json(), separators=(",", ":"),
    ).encode("utf-8")
    events_payload = json.dumps(build_event_sidecar(assessment), indent=2).encode("utf-8")
    topic_health_payload = json.dumps({
        "schema_version": 1,
        "run_id": run_id,
        "source": "research1_bag_metadata",
        "topic_message_counts": dict(sorted(evidence.topic_counts.items())),
        "missing_channels": assessment["missing_channels"],
        "protected_test_used": False,
    }, indent=2, sort_keys=True).encode("utf-8")
    summary = build_summary(
        catalog_row=row, aggregate=aggregate, route=route, assessment=assessment,
        bag_dir=bag_dir, bag_checksum_sha256=bag_checksum,
        aggregate_path=aggregate_path, aggregate_sha256=row["aggregate_sha256"],
        event_sidecar=outputs["events"], topic_health_sidecar=outputs["topic_health"],
        clock_map_path=outputs["clock_map"],
        clock_map_sha256=hashlib.sha256(clock_payload).hexdigest(),
        adapter_config_hash=adapter_config_hash(),
    )
    summary_payload = yaml.safe_dump(summary, sort_keys=False).encode("utf-8")
    publish_new_bytes(outputs["clock_map"], clock_payload)
    publish_new_bytes(outputs["events"], events_payload)
    publish_new_bytes(outputs["topic_health"], topic_health_payload)
    publish_new_bytes(outputs["summary"], summary_payload)
    return {
        "run_id": run_id,
        "summary": str(outputs["summary"]),
        "summary_sha256": hashlib.sha256(summary_payload).hexdigest(),
        "bag_checksum_sha256": bag_checksum,
        "primary_event_class": assessment["primary_event_class"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--run-id", action="append", required=True,
                        help="admitted Research 1 run_id (repeatable)")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    rows = {
        json.loads(line)["run_id"]: json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    thresholds = AdapterThresholds()
    failures = 0
    for run_id in args.run_id:
        row = rows.get(run_id)
        if row is None:
            print(f"{run_id}: not in the admitted adapter manifest; refusing", file=sys.stderr)
            failures += 1
            continue
        try:
            result = adapt(row, args.output_root, thresholds)
        except (ValueError, FileExistsError) as error:
            print(f"{run_id}: {error}", file=sys.stderr)
            failures += 1
            continue
        print(json.dumps(result, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

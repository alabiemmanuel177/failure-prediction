#!/usr/bin/env python3
"""Audit Research 1 structural reuse without reading protected episode outcomes."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
R1 = Path("/home/eao/risk-calibrated-nav")
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import sha256_file  # noqa: E402


def main() -> int:
    pilot = R1 / "manifests/pilot.csv"
    confirmatory = R1 / "manifests/confirmatory.csv"
    bag_root = R1 / "results/bags"
    pilot_rows = sum(1 for _ in pilot.open(encoding="utf-8")) - 1
    bag_directories = sum(path.is_dir() for path in bag_root.iterdir())
    bag_metadata = len(list(bag_root.glob("*/metadata.yaml")))
    with pilot.open(newline="", encoding="utf-8") as stream:
        manifest_fields = next(csv.reader(stream))
    # Read only the header of one development aggregate to establish schema. Never
    # inspect confirmatory outcomes or protected run records in this audit.
    development_example = R1 / "results/raw/6c110c1c-0d6f-4d47-91e0-3f62446c8ea0.csv"
    with development_example.open(newline="", encoding="utf-8") as stream:
        aggregate_fields = next(csv.reader(stream))
    run_maps: dict[str, str] = {}
    for path in (R1 / "results/raw").rglob("*.csv"):
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream), None)
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        if not row or not row.get("run_id") or not row.get("map_id"):
            continue
        run_id, map_id = str(row["run_id"]), str(row["map_id"])
        if run_id in run_maps and run_maps[run_id] != map_id:
            raise SystemExit(f"Research 1 run has conflicting map identities: {run_id}")
        run_maps[run_id] = map_id
    bag_split_counts: Counter[str] = Counter()
    compatible_by_split: Counter[str] = Counter()
    topic_compatible = 0
    eligible_bags = 0
    required_topics = {"/cmd_vel", "/odom", "/amcl_pose", "/scan", "/plan"}
    for bag_dir in bag_root.iterdir():
        if not bag_dir.is_dir():
            continue
        map_id = run_maps.get(bag_dir.name)
        split = (
            "development" if map_id and map_id.startswith("dev_")
            else "validation" if map_id and map_id.startswith("val_")
            else "protected_test" if map_id and map_id.startswith("test_")
            else "unresolved"
        )
        bag_split_counts[split] += 1
        if split not in {"development", "validation"}:
            continue
        eligible_bags += 1
        metadata_path = bag_dir / "metadata.yaml"
        if not metadata_path.exists():
            continue
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))[
            "rosbag2_bagfile_information"
        ]
        topics = {
            item["topic_metadata"]["name"]
            for item in metadata.get("topics_with_message_count", [])
            if int(item.get("message_count", 0)) > 0
        }
        topic_compatible += required_topics <= topics
        if required_topics <= topics:
            compatible_by_split[split] += 1
    report = {
        "schema_version": 1,
        "audit_type": "research1_structural_reuse_compatibility",
        "protected_outcomes_consulted": False,
        "research1_root": str(R1),
        "sources": {
            "pilot_manifest": str(pilot),
            "pilot_manifest_sha256": sha256_file(pilot),
            "confirmatory_manifest_sha256": sha256_file(confirmatory),
            "development_aggregate_schema_example": str(development_example),
            "development_aggregate_header_sha256": hashlib.sha256(
                (",".join(aggregate_fields) + "\n").encode("utf-8")
            ).hexdigest(),
        },
        "counts": {
            "research1_pilot_episode_rows": pilot_rows,
            "retained_bag_directories_all_research1_campaigns": bag_directories,
            "retained_bags_with_metadata": bag_metadata,
            "retained_bags_by_split_identity": dict(sorted(bag_split_counts.items())),
            "development_or_validation_bags": eligible_bags,
            "development_or_validation_bags_with_core_temporal_topics": topic_compatible,
            "core_temporal_topics_by_split": dict(sorted(compatible_by_split.items())),
            "research2_development_episodes": 648,
            "research2_targeted_development_episodes": 1212,
            "maximum_structurally_compatible_combined_episodes": 648 + topic_compatible,
            "maximum_fitting_episodes_preserving_split_independence": (
                648 + 1212 + compatible_by_split["development"]
            ),
            "validation_episodes_reserved_for_selection": compatible_by_split["validation"],
            "protocol_target_fitting_episodes": 3000,
        },
        "schema": {
            "pilot_manifest_fields": manifest_fields,
            "aggregate_episode_fields": aggregate_fields,
            "aggregate_is_time_series": False,
            "aggregate_has_decision_timestamp": False,
            "aggregate_has_injection_onset": False,
            "aggregate_has_research2_event_stream": False,
        },
        "reuse_decision": {
            "shared_platform_maps_routes_nav2_and_provenance": "eligible_under_content_lock",
            "episode_aggregate_rows_as_temporal_model_windows": "forbidden",
            "retained_development_or_validation_bags": (
                "structurally eligible only as natural or non-Research-2-fault episodes; "
                "requires a causal adapter and label audit after human admission"
            ),
            "research1_confirmatory_or_test_bags": "forbidden_for_fitting_and tuning",
            "reason": (
                "episode aggregates cannot supply past-only five-second telemetry windows; "
                "only split-safe bags with the complete temporal topic contract can be considered, "
                "and they do not contain Research 2's seven deterministic fault families"
            ),
        },
        "next_decision": (
            "after human admission, audit and adapt the structurally compatible Research 1 "
            "development/validation bags as a separate natural-failure/control source; keep "
            "Research 2 fault-family results stratified and evaluate learning curves before additions"
        ),
    }
    output = ROOT / "reports/integrity/research1_reuse_compatibility_v2.yaml"
    if output.exists():
        existing = yaml.safe_load(output.read_text(encoding="utf-8"))
        if existing != report:
            raise SystemExit(f"refusing to overwrite differing audit: {output}")
        print(f"Research 1 reuse audit already current: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
        print(f"wrote Research 1 reuse audit to {output}")
    print(
        f"R1 REUSE AUDIT: {pilot_rows} aggregate pilot rows; {bag_metadata} retained bags; "
        f"{topic_compatible} split-safe bags have core temporal topics"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

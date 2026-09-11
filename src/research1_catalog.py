"""Outcome-blind structural cataloging for retained Research 1 temporal bags."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.dataset_inventory import sha256_file


IDENTITY_FIELDS = ("run_id", "map_id", "route_id", "system_id", "seed")
FORBIDDEN_CATALOG_FIELDS = {
    "terminal_state", "success", "collision", "timeout", "invalid_reason",
    "duration_s", "spl", "goal_distance_gt_m", "shift_family", "severity",
}


def split_for_map(map_id: str) -> str | None:
    if map_id.startswith("dev_"):
        return "development"
    if map_id.startswith("val_"):
        return "validation"
    return None


def structural_identities(raw_root: Path) -> dict[str, dict[str, str]]:
    """Read identity columns only; outcome columns are deliberately never exported."""
    identities: dict[str, dict[str, str]] = {}
    for path in sorted(raw_root.rglob("*.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                row = next(reader, None)
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        if not row or not row.get("run_id") or not row.get("map_id"):
            continue
        identity = {
            field: str(row.get(field) or "")
            for field in IDENTITY_FIELDS
        }
        run_id = identity["run_id"]
        prior = identities.get(run_id)
        if prior is not None and prior != identity:
            raise ValueError(f"conflicting structural identity for Research 1 run {run_id}")
        identities[run_id] = identity
    return identities


def _bag_structure(metadata_path: Path) -> dict[str, Any]:
    document = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    info = document["rosbag2_bagfile_information"]
    topic_counts = {
        str(item["topic_metadata"]["name"]): int(item.get("message_count", 0))
        for item in info.get("topics_with_message_count", [])
    }
    return {
        "metadata_sha256": sha256_file(metadata_path),
        "message_count": int(info.get("message_count", 0)),
        "duration_nanoseconds": int(info.get("duration", {}).get("nanoseconds", 0)),
        "all_topic_count": len(topic_counts),
        "_topic_message_counts": topic_counts,
    }


def build_structural_catalog(
    research1_root: Path,
    *,
    required_topics: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    identities = structural_identities(research1_root / "results/raw")
    bag_root = research1_root / "results/bags"
    records: list[dict[str, Any]] = []
    counts = {
        "development_candidates": 0,
        "validation_candidates": 0,
        "missing_metadata": 0,
        "incompatible_topics": 0,
    }
    for bag_dir in sorted(path for path in bag_root.iterdir() if path.is_dir()):
        identity = identities.get(bag_dir.name)
        if identity is None:
            continue
        split = split_for_map(identity["map_id"])
        if split is None:
            continue
        counts[f"{split}_candidates"] += 1
        metadata_path = bag_dir / "metadata.yaml"
        if not metadata_path.is_file():
            counts["missing_metadata"] += 1
            continue
        structure = _bag_structure(metadata_path)
        topic_counts = structure.pop("_topic_message_counts")
        present = {
            topic for topic, count in topic_counts.items()
            if count > 0
        }
        if not required_topics <= present:
            counts["incompatible_topics"] += 1
            continue
        record = {
            **identity,
            "split": split,
            "bag_path": f"results/bags/{bag_dir.name}",
            "required_topics_present": sorted(required_topics),
            "required_topic_message_counts": {
                topic: topic_counts[topic] for topic in sorted(required_topics)
            },
            **structure,
            "admission_status": "structural_only_pending_human_and_causal_adapter_gates",
            "protected_outcomes_consulted": False,
        }
        leaked = FORBIDDEN_CATALOG_FIELDS & set(record)
        if leaked:
            raise ValueError(f"outcome or treatment fields entered structural catalog: {sorted(leaked)}")
        records.append(record)
    records.sort(key=lambda item: (item["split"], item["map_id"], item["route_id"], item["run_id"]))
    return records, counts


def validate_structural_catalog(
    records: list[dict[str, Any]],
    *,
    research1_root: Path,
    required_topics: set[str],
) -> list[str]:
    """Reconcile a published catalog against outcome-blind source metadata."""
    findings: list[str] = []
    run_ids = [str(item.get("run_id", "")) for item in records]
    duplicates = sorted(key for key, count in Counter(run_ids).items() if count > 1)
    if duplicates:
        findings.append(f"duplicate run ids: {duplicates[:5]}")
    expected_required = sorted(required_topics)
    for index, record in enumerate(records):
        run_id = str(record.get("run_id", ""))
        split = split_for_map(str(record.get("map_id", "")))
        if split is None or record.get("split") != split:
            findings.append(f"row {index} has forbidden or inconsistent split")
        if FORBIDDEN_CATALOG_FIELDS & set(record):
            findings.append(f"row {index} contains forbidden outcome/treatment fields")
        if record.get("required_topics_present") != expected_required:
            findings.append(f"row {index} has an unexpected required-topic declaration")
        counts = record.get("required_topic_message_counts")
        if not isinstance(counts, dict) or set(counts) != required_topics or any(
            not isinstance(value, int) or value <= 0 for value in (counts or {}).values()
        ):
            findings.append(f"row {index} has invalid required-topic counts")
        expected_relative = f"results/bags/{run_id}"
        if record.get("bag_path") != expected_relative:
            findings.append(f"row {index} has an unexpected bag path")
            continue
        metadata_path = research1_root / expected_relative / "metadata.yaml"
        if not metadata_path.is_file():
            findings.append(f"row {index} source metadata is missing")
            continue
        if record.get("metadata_sha256") != sha256_file(metadata_path):
            findings.append(f"row {index} source metadata checksum changed")
            continue
        structure = _bag_structure(metadata_path)
        source_counts = structure["_topic_message_counts"]
        expected_counts = {topic: source_counts.get(topic, 0) for topic in expected_required}
        if counts != expected_counts:
            findings.append(f"row {index} required-topic counts changed")
        if record.get("all_topic_count") != structure["all_topic_count"]:
            findings.append(f"row {index} topic cardinality changed")
        if record.get("message_count") != structure["message_count"]:
            findings.append(f"row {index} message count changed")
        if record.get("duration_nanoseconds") != structure["duration_nanoseconds"]:
            findings.append(f"row {index} duration changed")
        if record.get("protected_outcomes_consulted") is not False:
            findings.append(f"row {index} does not preserve the outcome-blind declaration")
        if record.get("admission_status") != (
            "structural_only_pending_human_and_causal_adapter_gates"
        ):
            findings.append(f"row {index} has an invalid admission state")
    return findings

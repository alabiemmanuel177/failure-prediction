#!/usr/bin/env python3
"""Publish an outcome-blind catalog of split-safe Research 1 temporal bags."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes, sha256_file  # noqa: E402
from src.research1_catalog import build_structural_catalog  # noqa: E402


RESEARCH1 = Path("/home/eao/risk-calibrated-nav")
REQUIRED_TOPICS = {"/cmd_vel", "/odom", "/amcl_pose", "/scan", "/plan"}


def main() -> int:
    catalog_path = ROOT / "data/manifests/research1_temporal_catalog_v2.jsonl"
    report_path = ROOT / "reports/integrity/research1_temporal_catalog_v2.yaml"
    records, exclusions = build_structural_catalog(
        RESEARCH1, required_topics=REQUIRED_TOPICS
    )
    payload = (
        "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in records)
    ).encode("utf-8")
    split_counts = {
        split: sum(item["split"] == split for item in records)
        for split in ("development", "validation")
    }
    report = {
        "schema_version": 2,
        "artifact_type": "research1_outcome_blind_temporal_structure_catalog",
        "research1_root": str(RESEARCH1),
        "required_topics": sorted(REQUIRED_TOPICS),
        "counts": {
            "catalog_rows": len(records),
            "by_split": split_counts,
            **exclusions,
        },
        "catalog": {
            "path": str(catalog_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "source_audit": {
            "path": "reports/integrity/research1_reuse_compatibility_v2.yaml",
            "sha256": sha256_file(
                ROOT / "reports/integrity/research1_reuse_compatibility_v2.yaml"
            ),
        },
        "controls": {
            "identity_columns_read": ["run_id", "map_id", "route_id", "system_id", "seed"],
            "outcome_columns_exported": [],
            "treatment_columns_exported": [],
            "nonrequired_topic_names_exported": False,
            "protected_test_rows_exported": 0,
            "bag_payloads_read": False,
            "admitted_for_model_use": False,
            "next_gate": "distinct human review then causal adapter and label audit",
        },
        "supersedes": {
            "artifact": "reports/integrity/research1_temporal_catalog_v1.yaml",
            "reason": "v1 exported non-required topic names from structural metadata; no outcome values were exported, but v2 minimizes the contract before adapter admission",
        },
    }
    report_payload = yaml.safe_dump(report, sort_keys=False).encode("utf-8")
    if catalog_path.exists() or report_path.exists():
        if not (
            catalog_path.is_file() and report_path.is_file()
            and catalog_path.read_bytes() == payload
            and report_path.read_bytes() == report_payload
        ):
            raise SystemExit("refusing to overwrite differing Research 1 structural catalog")
    else:
        publish_new_bytes(catalog_path, payload)
        publish_new_bytes(report_path, report_payload)
    expected = {"development": 825, "validation": 364}
    if split_counts != expected:
        raise SystemExit(f"unexpected split-safe Research 1 catalog counts: {split_counts}")
    print(
        f"RESEARCH 1 STRUCTURAL CATALOG PASS: {len(records)} rows; "
        f"development={split_counts['development']}, validation={split_counts['validation']}; "
        "0 protected rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

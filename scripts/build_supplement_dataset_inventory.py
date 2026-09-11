#!/usr/bin/env python3
"""Build the immutable development-supplement inventory (size from its manifest)."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import (  # noqa: E402
    episode_record, inventory_report, jsonl_bytes, publish_new_bytes,
    select_scientific_summaries, sha256_file,
)
from src.experiments import expand_balanced_pilot  # noqa: E402


def main() -> int:
    campaign_path = ROOT / "data/manifests/development_supplement_v1.yaml"
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    expected_count = int(campaign["expected_episode_count"])
    result_path = ROOT / f"reports/pilot/development_supplement_v1.cumulative{expected_count}.yaml"
    inventory_path = ROOT / "data/manifests/development_supplement_v1.episodes.jsonl"
    output_path = ROOT / "data/manifests/development_supplement_v1.dataset.yaml"
    if inventory_path.exists() or output_path.exists():
        raise SystemExit("refusing to overwrite an existing supplement inventory")
    if not result_path.exists():
        raise SystemExit("development supplement campaign is not complete")
    result = yaml.safe_load(result_path.read_text(encoding="utf-8"))
    if not result.get("complete_and_artifact_valid") or result.get("protected_test_used"):
        raise SystemExit("supplement report is incomplete, invalid, or protected")
    summaries = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted((ROOT / "data/raw/summaries").glob("*.yaml"))
    ]
    expected = expand_balanced_pilot(campaign)
    selected = select_scientific_summaries(
        campaign["campaign_id"], [item["episode_key"] for item in expected],
        summaries, campaign.get("infrastructure_replacements", []),
    )
    records = sorted(
        (episode_record(ROOT, key, summary, replacement)
         for key, summary, replacement in selected),
        key=lambda item: item["dataset_episode_key"],
    )
    payload = jsonl_bytes(records)
    report = inventory_report(
        records,
        dataset_id=f"development_supplement_v1-development-{expected_count}",
        source_hashes={
            "campaign_manifest_sha256": sha256_file(campaign_path),
            "pilot_report_sha256": sha256_file(result_path),
        },
        inventory_sha256=hashlib.sha256(payload).hexdigest(),
        inventory_path="data/manifests/development_supplement_v1.episodes.jsonl",
        collection_status="development_supplement_complete",
    )
    if report["counts"]["scientific_episodes"] != expected_count:
        raise SystemExit(f"resolved inventory does not contain exactly {expected_count} episodes")
    publish_new_bytes(inventory_path, payload)
    publish_new_bytes(output_path, yaml.safe_dump(report, sort_keys=False).encode("utf-8"))
    print(f"wrote {len(records)} development-supplement rows to {inventory_path}")
    print(f"wrote hash-addressed supplement manifest to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

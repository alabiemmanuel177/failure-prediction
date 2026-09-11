#!/usr/bin/env python3
"""Build the immutable, non-model 648-episode development inventory."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", type=Path,
        default=ROOT / "data/manifests/balanced_pilot_v1.yaml",
    )
    parser.add_argument(
        "--pilot-report", type=Path,
        default=ROOT / "reports/pilot/balanced_pilot_v1.cumulative648.yaml",
    )
    parser.add_argument(
        "--inventory", type=Path,
        default=ROOT / "data/manifests/balanced_pilot_v1.episodes.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/manifests/balanced_pilot_v1.dataset.yaml",
    )
    args = parser.parse_args()
    if args.inventory.exists() or args.output.exists():
        raise SystemExit("refusing to overwrite an existing dataset inventory")
    campaign = yaml.safe_load(args.campaign.read_text(encoding="utf-8"))
    pilot = yaml.safe_load(args.pilot_report.read_text(encoding="utf-8"))
    if not pilot.get("complete_and_artifact_valid") or pilot.get("protected_test_used"):
        raise SystemExit("pilot report is incomplete, invalid, or protected")
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
    inventory_sha256 = hashlib.sha256(payload).hexdigest()
    report = inventory_report(
        records,
        dataset_id="balanced_pilot_v1-development-648",
        source_hashes={
            "campaign_manifest_sha256": sha256_file(args.campaign),
            "pilot_report_sha256": sha256_file(args.pilot_report),
        },
        inventory_sha256=inventory_sha256,
    )
    if report["counts"]["scientific_episodes"] != 648:
        raise SystemExit("resolved inventory does not contain exactly 648 scientific episodes")
    publish_new_bytes(args.inventory, payload)
    publish_new_bytes(args.output, yaml.safe_dump(report, sort_keys=False).encode("utf-8"))
    print(f"wrote {len(records)} episode rows to {args.inventory}")
    print(f"wrote hash-addressed dataset manifest to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

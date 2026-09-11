#!/usr/bin/env python3
"""Build a hash-addressed episode inventory for any completed non-protected campaign.

Generalises the per-campaign inventory builders: the campaign manifest supplies the
design and expected count, the cumulative report proves completion, and the output
follows the same row schema as `balanced_pilot_v1.episodes.jsonl` so
`scripts/extract_dataset_sequences.py` can derive it. Refuses protected campaigns.
"""

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
from src.experiments import campaign_episodes  # noqa: E402
from src.experiments.campaigns import declared_replacements  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="cumulative campaign report")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--collection-status", default="collection_complete")
    parser.add_argument("--allow-protected-after-freeze", action="store_true")
    args = parser.parse_args()
    campaign = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    protected = campaign.get("protected_test_used")
    allow_protected = False
    if protected is True:
        import subprocess
        gate = subprocess.run([sys.executable, str(ROOT / "scripts/check_readiness.py"), "--stage", "confirmatory"],
                              check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not (args.allow_protected_after_freeze and gate):
            raise SystemExit("protected campaign inventory requires --allow-protected-after-freeze and a passing confirmatory gate")
        allow_protected = True
    elif protected is not False:
        raise SystemExit("campaign manifest must declare protected_test_used")
    expected_count = int(campaign["expected_episode_count"])
    campaign_id = campaign["campaign_id"]
    inventory_path = ROOT / "data/manifests" / f"{campaign_id}.episodes.jsonl"
    output_path = ROOT / "data/manifests" / f"{campaign_id}.dataset.yaml"
    if inventory_path.exists() or output_path.exists():
        raise SystemExit(f"refusing to overwrite an existing inventory for {campaign_id}")
    result = yaml.safe_load(args.report.read_text(encoding="utf-8"))
    counts = result.get("counts", {})
    if not (result.get("complete_and_artifact_valid") is True and result.get("protected_test_used") is protected
            and counts.get("expected") == counts.get("usable") == expected_count):
        raise SystemExit("campaign report is incomplete, invalid or protected")
    summaries = [yaml.safe_load(path.read_text(encoding="utf-8"))
                 for path in sorted((ROOT / "data/raw/summaries").glob("*.yaml"))]
    expected = campaign_episodes(campaign)
    selected = select_scientific_summaries(
        campaign_id, [item["episode_key"] for item in expected], summaries,
        declared_replacements(ROOT, campaign), allow_protected=allow_protected,
    )
    records = sorted((episode_record(ROOT, key, summary, replacement)
                      for key, summary, replacement in selected),
                     key=lambda item: item["dataset_episode_key"])
    payload = jsonl_bytes(records)
    report = inventory_report(
        records, dataset_id=args.dataset_id,
        source_hashes={"campaign_manifest_sha256": sha256_file(args.manifest),
                       "pilot_report_sha256": sha256_file(args.report)},
        inventory_sha256=hashlib.sha256(payload).hexdigest(),
        inventory_path=str(inventory_path.relative_to(ROOT)),
        collection_status=args.collection_status, allow_protected=allow_protected,
    )
    if report["counts"]["scientific_episodes"] != expected_count:
        raise SystemExit(f"resolved inventory does not contain exactly {expected_count} episodes")
    publish_new_bytes(inventory_path, payload)
    publish_new_bytes(output_path, yaml.safe_dump(report, sort_keys=False).encode("utf-8"))
    print(f"wrote {len(records)} rows to {inventory_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

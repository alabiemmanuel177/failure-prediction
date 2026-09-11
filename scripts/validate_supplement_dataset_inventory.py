#!/usr/bin/env python3
"""Validate the hash-addressed development-supplement inventory."""

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import validate_inventory  # noqa: E402


def main() -> int:
    document = yaml.safe_load(
        (ROOT / "data/manifests/development_supplement_v1.dataset.yaml").read_text(
            encoding="utf-8"
        )
    )
    campaign = yaml.safe_load(
        (ROOT / "data/manifests/development_supplement_v1.yaml").read_text(encoding="utf-8")
    )
    expected_count = int(campaign["expected_episode_count"])
    findings = validate_inventory(
        ROOT, document, expected_split="development",
        source_paths={
            "campaign_manifest_sha256": ROOT / "data/manifests/development_supplement_v1.yaml",
            "pilot_report_sha256": ROOT / f"reports/pilot/development_supplement_v1.cumulative{expected_count}.yaml",
        },
    )
    if findings:
        print(f"SUPPLEMENT DATASET INVALID: {len(findings)} finding(s)")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(f"SUPPLEMENT DATASET VALID: {expected_count} unique development episodes; hashes and sidecars agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

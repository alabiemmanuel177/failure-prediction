#!/usr/bin/env python3
"""Validate the hash-addressed targeted-development inventory."""

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import validate_inventory  # noqa: E402


def main() -> int:
    document = yaml.safe_load(
        (ROOT / "data/manifests/targeted_development_v1.dataset.yaml").read_text(
            encoding="utf-8"
        )
    )
    findings = validate_inventory(
        ROOT, document, expected_split="development",
        source_paths={
            "campaign_manifest_sha256": ROOT / "data/manifests/targeted_development_v1.yaml",
            "pilot_report_sha256": ROOT / "reports/pilot/targeted_development_v1.cumulative1212.yaml",
        },
    )
    if findings:
        print(f"TARGETED DATASET INVALID: {len(findings)} finding(s)")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("TARGETED DATASET VALID: 1,212 unique development episodes; hashes and sidecars agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

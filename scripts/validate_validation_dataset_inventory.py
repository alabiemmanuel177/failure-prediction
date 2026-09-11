#!/usr/bin/env python3
"""Validate the hash-addressed 324-episode validation inventory."""

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import validate_inventory  # noqa: E402


def main() -> int:
    document = yaml.safe_load(
        (ROOT / "data/manifests/balanced_validation_v1.dataset.yaml").read_text(
            encoding="utf-8"
        )
    )
    findings = validate_inventory(
        ROOT, document, expected_split="validation",
        source_paths={
            "campaign_manifest_sha256": ROOT / "data/manifests/balanced_validation_v1.yaml",
            "pilot_report_sha256": ROOT / "reports/validation/balanced_validation_v1.cumulative324.yaml",
        },
    )
    if findings:
        print(f"VALIDATION DATASET INVALID: {len(findings)} finding(s)")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("VALIDATION DATASET VALID: 324 unique validation episodes; hashes and sidecars agree")
    print("selection admission remains governed by scripts/check_readiness.py --stage training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

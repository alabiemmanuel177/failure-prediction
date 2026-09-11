#!/usr/bin/env python3
"""Validate the hash-addressed 648-episode development inventory."""

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import validate_inventory  # noqa: E402


def main() -> int:
    path = ROOT / "data/manifests/balanced_pilot_v1.dataset.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    findings = validate_inventory(ROOT, document)
    if findings:
        print(f"DEVELOPMENT DATASET INVALID: {len(findings)} finding(s)")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("DEVELOPMENT DATASET VALID: 648 unique development episodes; hashes and sidecars agree")
    print("training admission remains governed by scripts/check_readiness.py --stage training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate a completed supervisor threshold review without applying it."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labels.threshold_review import validate_threshold_review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    args = parser.parse_args()
    document = yaml.safe_load(args.review.read_text(encoding="utf-8"))
    researcher = yaml.safe_load(
        (ROOT / "configs/event_threshold_review.2026-08-30.researcher.yaml")
        .read_text(encoding="utf-8")
    )
    findings = validate_threshold_review(
        document, researcher["decisions"],
        researcher_name=str(researcher["reviewer_name"])
    )
    if findings:
        print(f"REVIEW INVALID: {len(findings)} finding(s)")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("REVIEW VALID: decisions are complete; values have not been applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

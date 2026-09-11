#!/usr/bin/env python3
"""Validate one per-episode sequence artifact against the frozen feature order."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    load_primary_feature_set, model_columns, validate_sequence_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--allow-protected", action="store_true")
    args = parser.parse_args()
    primary = load_primary_feature_set(ROOT / "configs/feature_schema.yaml")
    findings = validate_sequence_artifact(
        args.artifact,
        expected_feature_names=model_columns(primary),
        allow_protected=args.allow_protected,
    )
    if findings:
        print(f"INVALID sequence artifact: {len(findings)} finding(s)")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(f"VALID sequence artifact: {args.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

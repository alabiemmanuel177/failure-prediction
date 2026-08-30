#!/usr/bin/env python3
"""Validate one episode annotation against the frozen failure taxonomy."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labels.annotations import validate_annotation


def load(path: Path):
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation", type=Path)
    args = parser.parse_args()
    errors = validate_annotation(
        load(args.annotation), load(ROOT / "configs" / "failure_taxonomy.yaml")
    )
    if errors:
        print(f"INVALID: {len(errors)} finding(s)")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"VALID: {args.annotation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

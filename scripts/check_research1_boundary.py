#!/usr/bin/env python3
"""Check Research 1 against the G6 content boundary without launching ROS."""

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.platform_boundary import BoundaryError, validate_platform


def main() -> int:
    lock = yaml.safe_load((ROOT / "integration" / "research1.lock.yaml").read_text())
    try:
        research1, actual_commit = validate_platform(lock)
    except (BoundaryError, OSError, KeyError, yaml.YAMLError) as error:
        print(f"BOUNDARY FAIL: {error}")
        return 1
    print(f"BOUNDARY PASS: {research1}")
    print(f"platform_commit={lock['commit_sha']}")
    print(f"repository_head={actual_commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

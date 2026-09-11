#!/usr/bin/env python3
"""Apply a fitted calibrator to a prediction table, writing a new immutable table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import sha256_file
from src.evaluation.calibrators import Calibrator, apply_calibrator_rows
from src.evaluation.prediction_tables import (
    guard_protected_rows, read_prediction_table, write_prediction_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="uncalibrated prediction table")
    parser.add_argument("calibrator", type=Path, help="calibrator JSON from select_calibration.py")
    parser.add_argument("output", type=Path, help="new prediction table (must not exist)")
    parser.add_argument(
        "--allow-protected-after-freeze", action="store_true",
        help="permit held-out rows; still requires check_readiness --stage confirmatory",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    rows = read_prediction_table(args.predictions)
    try:
        protected = guard_protected_rows(
            rows, explicitly_allowed=args.allow_protected_after_freeze, root=ROOT
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    already = [
        row for row in rows if float(row["risk_score"]) != float(row["raw_score"])
    ]
    if already:
        raise SystemExit(
            "input table already carries calibrated risk_score; calibrators are applied "
            "to raw_score of an uncalibrated table exactly once"
        )
    calibrator = Calibrator.from_json_bytes(args.calibrator.read_bytes())
    calibrated = apply_calibrator_rows(rows, calibrator)
    digest = write_prediction_table(args.output, calibrated)
    print(
        f"applied {calibrator.method} (sha256 {calibrator.sha256()[:12]}) to "
        f"{args.predictions} (sha256 {sha256_file(args.predictions)[:12]}) -> {args.output} "
        f"(sha256 {digest[:12]}, protected_test_used={'true' if protected else 'false'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

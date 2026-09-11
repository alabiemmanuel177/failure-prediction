#!/usr/bin/env python3
"""Wait for every frozen-fitting-pool dataset to be derived, then run the final pass.

Polls for the all-decisions report of each development dataset named in Protocol
Amendment PA-2026-09-03-03, then runs `scripts/run_model_development.py --tag final_v1`.
It performs validation-only selection; the model freeze remains a separate,
researcher-signed step.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, default=ROOT / "configs/protocol_amendment_1.3.yaml")
    parser.add_argument("--tag", default="final_v1")
    parser.add_argument("--poll-seconds", type=int, default=600)
    args = parser.parse_args()
    amendment = yaml.safe_load(args.amendment.read_text(encoding="utf-8"))
    datasets = list(amendment["decisions"]["frozen_fitting_pool"])
    selection = amendment["decisions"]["selection_dataset"]
    while True:
        missing = [d for d in datasets + [selection]
                   if not (ROOT / "data/derived" / d / "decisions_report.yaml").exists()]
        if not missing:
            break
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} waiting for derived datasets: {missing}", flush=True)
        time.sleep(args.poll_seconds)
    if (ROOT / "reports/model_selection" / args.tag / "validation_comparison.yaml").exists():
        print(f"final pass {args.tag} already complete")
        return 0
    command = [sys.executable, str(ROOT / "scripts/run_model_development.py"), "--tag", args.tag,
               "--selection-dataset", selection]
    for dataset in datasets:
        command += ["--train-dataset", dataset]
    print("$ " + " ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate an immutable episode-level report for one completed pilot wave."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import summarize_pilot_wave


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/manifests/balanced_pilot_v1.yaml",
    )
    parser.add_argument("--summary-root", type=Path, default=ROOT / "data/raw/summaries")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    campaign_id = str(manifest["campaign_id"])
    ledger_path = args.ledger or ROOT / "logs/campaigns" / f"{campaign_id}.jsonl"
    output = args.output or ROOT / "reports/pilot" / f"{campaign_id}.wave{args.wave}.yaml"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    summaries = [yaml.safe_load(path.read_text(encoding="utf-8"))
                 for path in sorted(args.summary_root.glob("*.yaml"))]
    report = summarize_pilot_wave(
        campaign_id=campaign_id,
        wave=args.wave,
        ledger_rows=rows,
        summaries=summaries,
        episodes_per_wave=int(
            manifest["execution_policy"]["maximum_episodes_per_invocation"]
        ),
        replacement_specs=manifest.get("infrastructure_replacements", ()),
    )
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)
    print(f"wrote pilot wave report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

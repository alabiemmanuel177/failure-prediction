#!/usr/bin/env python3
"""Create the preregistered episode-level pilot inventory from retained summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import declared_replacements, campaign_episodes, ledger_invalid_episode_keys, summarize_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/manifests/balanced_pilot_v1.yaml",
    )
    parser.add_argument("--summary-root", type=Path, default=ROOT / "data/raw/summaries")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/pilot/balanced_pilot_v1.yaml",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    summaries = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted(args.summary_root.glob("*.yaml"))
    ]
    ledger_invalid = ledger_invalid_episode_keys(ROOT, manifest)
    report = summarize_pilot(
        manifest["campaign_id"], campaign_episodes(manifest), summaries,
        declared_replacements(ROOT, manifest), ledger_invalid_keys=ledger_invalid,
    )
    if not report["complete_and_artifact_valid"] and not args.allow_incomplete:
        raise SystemExit(
            "pilot is incomplete or contains invalid artifacts; use --allow-incomplete "
            "only for operational progress reporting"
        )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(f"wrote episode-level pilot inventory to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

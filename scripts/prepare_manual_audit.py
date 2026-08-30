#!/usr/bin/env python3
"""Create review-pending annotations for the predeclared 20-episode audit sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def select_audit_keys(
    primary: list[str],
    reserves: list[str],
    by_key: dict[str, tuple[Path, dict]],
    infrastructure_invalid: set[str],
) -> list[str]:
    """Apply the frozen primary/reserve order, including pre-summary failures."""
    selected: list[str] = []
    invalid_count = 0
    for key in primary:
        if key in infrastructure_invalid:
            invalid_count += 1
            continue
        if key not in by_key:
            raise SystemExit(f"campaign summary missing for primary audit key: {key}")
        if by_key[key][1].get("outcome", {}).get("terminal_state") == "invalid":
            invalid_count += 1
        else:
            selected.append(key)
    replacements = 0
    for key in reserves:
        if replacements >= invalid_count:
            break
        if key in infrastructure_invalid:
            continue
        if key not in by_key:
            raise SystemExit(f"campaign summary missing for reserve audit key: {key}")
        if by_key[key][1].get("outcome", {}).get("terminal_state") == "invalid":
            continue
        selected.append(key)
        replacements += 1
    if replacements != invalid_count:
        raise SystemExit("not enough valid prespecified reserves for infrastructure-invalid primaries")
    if len(selected) != 20:
        raise SystemExit(f"audit selection has {len(selected)} episodes, expected 20")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data" / "manifests" / "live_integrity_v1.yaml",
    )
    parser.add_argument("--summary-root", type=Path, default=ROOT / "data/raw/summaries")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/annotations")
    args = parser.parse_args()
    document = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    campaign_id = document["campaign_id"]
    ledger = ROOT / "logs" / "campaigns" / f"{campaign_id}.jsonl"
    primary = list(document["audit_sampling"]["primary_episode_keys"])
    reserves = list(document["audit_sampling"]["infrastructure_invalid_reserve_keys"])
    by_key: dict[str, tuple[Path, dict]] = {}
    for path in sorted(args.summary_root.glob("*.yaml")):
        summary = yaml.safe_load(path.read_text(encoding="utf-8"))
        identity = summary.get("identity", {})
        if identity.get("campaign_id") != campaign_id:
            continue
        key = identity.get("episode_key")
        if key in by_key:
            raise SystemExit(f"duplicate campaign episode key in summaries: {key}")
        by_key[key] = (path, summary)

    infrastructure_invalid: set[str] = set()
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = record["episode_key"]
            if int(record["returncode"]) != 0 and key not in by_key:
                infrastructure_invalid.add(key)
    exclusions_path = ROOT / "data" / "manifests" / f"{campaign_id}_infrastructure_invalid.yaml"
    if exclusions_path.exists():
        exclusions = yaml.safe_load(exclusions_path.read_text(encoding="utf-8"))
        infrastructure_invalid.update(
            item["episode_key"] for item in exclusions.get("episodes", [])
        )
    selected = select_audit_keys(primary, reserves, by_key, infrastructure_invalid)

    args.output_root.mkdir(parents=True, exist_ok=True)
    for key in selected:
        summary_path, summary = by_key[key]
        run_id = summary["identity"]["run_id"]
        output = args.output_root / f"{run_id}.yaml"
        if output.exists():
            raise SystemExit(f"refusing to overwrite annotation: {output}")
        validation = subprocess.run(
            [sys.executable, str(ROOT / "scripts/validate_episode_artifacts.py"),
             str(summary_path)], check=False,
        )
        if validation.returncode:
            raise SystemExit(f"artifact validation failed for {key}")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/extract_episode_annotation.py"),
             str(summary_path), str(output)], check=True,
        )
    print(f"prepared {len(selected)} review-pending annotations; human review still required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

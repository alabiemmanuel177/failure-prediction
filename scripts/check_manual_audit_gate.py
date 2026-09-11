#!/usr/bin/env python3
"""Check all 20 frozen audit episodes for exact primary-review label agreement."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_manual_audit import select_audit_keys  # noqa: E402
from src.labels.audit_gate import validate_primary_review  # noqa: E402


def load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def campaign_summaries(paths: list[Path], campaign_id: str) -> dict[str, tuple[Path, dict]]:
    selected: dict[str, tuple[Path, dict]] = {}
    for path in paths:
        summary = load(path)
        # Infrastructure-invalid payloads remain on disk by policy. They must not
        # crash a scientific scanner or masquerade as an episode mapping.
        if not isinstance(summary, dict):
            continue
        identity = summary.get("identity", {})
        if identity.get("campaign_id") == campaign_id:
            selected[str(identity["episode_key"])] = (path, summary)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/manifests/live_integrity_v3.yaml",
    )
    parser.add_argument("--automatic-root", type=Path, default=ROOT / "data/annotations")
    parser.add_argument("--reviewed-root", type=Path, default=ROOT / "data/annotations/reviewed")
    args = parser.parse_args()
    manifest = load(args.manifest)
    campaign_id = manifest["campaign_id"]
    by_key = campaign_summaries(
        list((ROOT / "data/raw/summaries").glob("*.yaml")), campaign_id
    )
    exclusions_path = ROOT / "data/manifests" / f"{campaign_id}_infrastructure_invalid.yaml"
    invalid = {
        item["episode_key"] for item in load(exclusions_path).get("episodes", [])
    } if exclusions_path.exists() else set()
    selected = select_audit_keys(
        manifest["audit_sampling"]["primary_episode_keys"],
        manifest["audit_sampling"]["infrastructure_invalid_reserve_keys"],
        by_key,
        invalid,
    )
    taxonomy = load(ROOT / "configs/failure_taxonomy.yaml")
    findings = []
    for key in selected:
        run_id = by_key[key][1]["identity"]["run_id"]
        automatic_path = args.automatic_root / f"{run_id}.yaml"
        reviewed_path = args.reviewed_root / f"{run_id}.yaml"
        if not automatic_path.exists():
            findings.append(f"{key}: automatic annotation missing")
            continue
        if not reviewed_path.exists():
            findings.append(f"{key}: primary reviewed annotation missing")
            continue
        automatic = load(automatic_path)
        reviewed = load(reviewed_path)
        if not isinstance(automatic, dict):
            findings.append(f"{key}: automatic annotation is not a YAML mapping")
            continue
        if not isinstance(reviewed, dict):
            findings.append(f"{key}: primary reviewed annotation is not a YAML mapping")
            continue
        for error in validate_primary_review(automatic, reviewed, taxonomy):
            findings.append(f"{key}: {error}")
    if findings:
        print(f"MANUAL AUDIT PENDING: {len(findings)} finding(s) across {len(selected)} frozen episodes")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(
        f"MANUAL AUDIT PASSED: exact primary-review agreement on "
        f"{len(selected)}/20 frozen episodes; no inter-rater agreement claimed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Finalize one complete pre-model campaign into a validated immutable inventory."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CollectionSpec:
    stage: str
    report: Path
    inventory: Path
    dataset_manifest: Path
    builder: Path
    validator: Path
    expected: int


def collection_spec(stage: str, root: Path = ROOT) -> CollectionSpec:
    values = {
        "validation": CollectionSpec(
            stage="validation",
            report=root / "reports/validation/balanced_validation_v1.cumulative324.yaml",
            inventory=root / "data/manifests/balanced_validation_v1.episodes.jsonl",
            dataset_manifest=root / "data/manifests/balanced_validation_v1.dataset.yaml",
            builder=root / "scripts/build_validation_dataset_inventory.py",
            validator=root / "scripts/validate_validation_dataset_inventory.py",
            expected=324,
        ),
        "targeted": CollectionSpec(
            stage="targeted",
            report=root / "reports/pilot/targeted_development_v1.cumulative1212.yaml",
            inventory=root / "data/manifests/targeted_development_v1.episodes.jsonl",
            dataset_manifest=root / "data/manifests/targeted_development_v1.dataset.yaml",
            builder=root / "scripts/build_targeted_dataset_inventory.py",
            validator=root / "scripts/validate_targeted_dataset_inventory.py",
            expected=1212,
        ),
    }
    supplement_manifest = root / "data/manifests/development_supplement_v1.yaml"
    if supplement_manifest.exists():
        supplement_count = int(
            yaml.safe_load(supplement_manifest.read_text(encoding="utf-8"))["expected_episode_count"]
        )
        values["supplement"] = CollectionSpec(
            stage="supplement",
            report=root / f"reports/pilot/development_supplement_v1.cumulative{supplement_count}.yaml",
            inventory=root / "data/manifests/development_supplement_v1.episodes.jsonl",
            dataset_manifest=root / "data/manifests/development_supplement_v1.dataset.yaml",
            builder=root / "scripts/build_supplement_dataset_inventory.py",
            validator=root / "scripts/validate_supplement_dataset_inventory.py",
            expected=supplement_count,
        )
    if stage not in values:
        raise ValueError(f"unknown non-model collection stage: {stage}")
    return values[stage]


def report_is_complete(spec: CollectionSpec) -> bool:
    if not spec.report.is_file():
        return False
    value = yaml.safe_load(spec.report.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return False
    counts = value.get("counts", {})
    integrity = value.get("integrity", {})
    return bool(
        value.get("complete_and_artifact_valid") is True
        and value.get("protected_test_used") is False
        and counts.get("expected") == spec.expected
        and counts.get("observed") == spec.expected
        and counts.get("usable") == spec.expected
        and all(
            integrity.get(name) == []
            for name in (
                "missing_episode_keys", "unexpected_episode_keys",
                "duplicate_episode_keys", "replacement_errors",
            )
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validation", "targeted", "supplement"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    spec = collection_spec(args.stage)
    if not report_is_complete(spec):
        raise SystemExit(f"{args.stage} campaign report is absent or not artifact-valid")
    inventory_exists = spec.inventory.exists()
    manifest_exists = spec.dataset_manifest.exists()
    if inventory_exists != manifest_exists:
        raise SystemExit("partial inventory publication requires integrity review")
    if args.dry_run:
        print(
            f"READY to finalize {args.stage}: expected={spec.expected} "
            f"already_published={inventory_exists}"
        )
        return 0
    if not inventory_exists:
        subprocess.run([sys.executable, str(spec.builder)], check=True)
    subprocess.run([sys.executable, str(spec.validator)], check=True)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/audit_raw_artifact_payloads.py")
    ], check=True)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/research_log.py"), "verify"
    ], check=True)
    print(f"FINALIZED {args.stage}: immutable inventory validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

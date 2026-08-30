#!/usr/bin/env python3
"""Fail closed when protocol configuration is not ready for data extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = (
    ROOT / "configs" / "failure_events.yaml",
    ROOT / "configs" / "failure_taxonomy.yaml",
    ROOT / "configs" / "feature_schema.yaml",
    ROOT / "configs" / "leakage_denylist.yaml",
    ROOT / "configs" / "alarm_policy.yaml",
    ROOT / "configs" / "recording.yaml",
    ROOT / "data" / "manifests" / "splits.template.yaml",
)


def walk(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, f"{path}[{index}]")
    else:
        yield path, value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be a mapping")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("draft", "extraction", "training", "confirmatory"),
        default="draft",
    )
    args = parser.parse_args()

    findings: list[str] = []
    documents: dict[Path, dict[str, Any]] = {}
    for path in CONFIG_FILES:
        try:
            documents[path] = load_yaml(path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            findings.append(str(error))

    for path, document in documents.items():
        for field, value in walk(document):
            if isinstance(value, str) and value.startswith("TODO"):
                findings.append(f"{path.relative_to(ROOT)}:{field} is {value}")

    if args.stage in {"extraction", "training", "confirmatory"}:
        for path in sorted((ROOT / "configs" / "faults").glob("*.yaml")):
            try:
                fault = load_yaml(path)
            except (OSError, ValueError, yaml.YAMLError) as error:
                findings.append(str(error))
                continue
            if fault.get("status") != "frozen":
                findings.append(
                    f"{path.relative_to(ROOT)}:status is {fault.get('status')}; "
                    "fault-integrity smoke and review are required before dataset extraction"
                )

    splits = documents.get(CONFIG_FILES[-1], {})
    if args.stage in {"extraction", "training", "confirmatory"}:
        required_splits = ["development", "validation"]
        if args.stage == "confirmatory":
            required_splits.append("held_out_map_test")
        for split_name in required_splits:
            split = splits.get(split_name, {})
            if not split.get("maps"):
                findings.append(f"splits:{split_name}.maps is empty")
            if not split.get("routes"):
                findings.append(f"splits:{split_name}.routes is empty")

    alarm = documents.get(ROOT / "configs" / "alarm_policy.yaml", {})
    if args.stage in {"training", "confirmatory"} and alarm.get("threshold") is None:
        findings.append("alarm_policy:threshold is not frozen")

    if findings:
        print(f"NOT READY for {args.stage}: {len(findings)} finding(s)")
        for finding in findings:
            print(f"- {finding}")
        return 1

    print(f"READY for {args.stage}: configuration checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

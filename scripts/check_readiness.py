#!/usr/bin/env python3
"""Fail closed when protocol configuration is not ready for data extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labels.audit_gate import validate_primary_review
from src.labels.threshold_review import validate_researcher_threshold_review

RESEARCHER_REVIEW = ROOT / "configs" / "event_threshold_review.2026-08-30.researcher.yaml"
PROTOCOL_AMENDMENT = ROOT / "configs" / "protocol_amendment_1.1.yaml"
MODEL_FREEZE = ROOT / "configs" / "model_freeze.yaml"
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
    if args.stage in {"training", "confirmatory"}:
        try:
            amendment = load_yaml(PROTOCOL_AMENDMENT)
            if amendment.get("status") != "approved":
                findings.append("Protocol 1.1 human-review amendment is not approved")
            if amendment.get("protected_outcomes_consulted") is not False:
                findings.append("Protocol 1.1 amendment must precede protected-outcome inspection")
            review = load_yaml(RESEARCHER_REVIEW)
            findings.extend(validate_researcher_threshold_review(review))
        except (OSError, ValueError, yaml.YAMLError) as error:
            findings.append(str(error))

        taxonomy = documents.get(ROOT / "configs" / "failure_taxonomy.yaml", {})
        automatic = sorted(
            path for path in (ROOT / "data/annotations").glob("*.yaml")
            if path.name != "episode.template.yaml"
        )
        if len(automatic) != 20:
            findings.append(f"manual audit baseline contains {len(automatic)} episodes, expected 20")
        for path in automatic:
            reviewed = ROOT / "data/annotations/reviewed" / path.name
            if not reviewed.exists():
                findings.append(f"independent reviewed annotation missing: {path.stem}")
                continue
            try:
                for error in validate_primary_review(
                    load_yaml(path), load_yaml(reviewed), taxonomy
                ):
                    findings.append(f"manual audit {path.stem}: {error}")
            except (OSError, ValueError, yaml.YAMLError) as error:
                findings.append(str(error))

    if args.stage == "confirmatory":
        if alarm.get("threshold") is None:
            findings.append("alarm_policy:threshold is not frozen")
        if not MODEL_FREEZE.exists():
            findings.append("model freeze missing: configs/model_freeze.yaml")
        else:
            try:
                freeze = load_yaml(MODEL_FREEZE)
                if freeze.get("frozen") is not True:
                    findings.append("model_freeze:frozen is not true")
                if freeze.get("protected_outcomes_consulted") is not False:
                    findings.append("model freeze must precede protected-outcome inspection")
                declaration = freeze.get("declaration", {})
                for field in (
                    "model_selection_complete", "calibration_selection_complete",
                    "threshold_selection_complete",
                ):
                    if declaration.get(field) is not True:
                        findings.append(f"model_freeze:declaration.{field} is not true")
                if declaration.get("protected_maps_or_outcomes_inspected") is not False:
                    findings.append(
                        "model_freeze: protected maps or outcomes must be uninspected"
                    )
                for field, value in walk(freeze):
                    if isinstance(value, str) and value.startswith("TODO"):
                        findings.append(f"configs/model_freeze.yaml:{field} is {value}")
            except (OSError, ValueError, yaml.YAMLError) as error:
                findings.append(str(error))

    if findings:
        print(f"NOT READY for {args.stage}: {len(findings)} finding(s)")
        for finding in findings:
            print(f"- {finding}")
        return 1

    print(f"READY for {args.stage}: configuration checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

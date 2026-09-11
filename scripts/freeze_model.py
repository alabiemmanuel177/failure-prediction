#!/usr/bin/env python3
"""Freeze the primary predictor, calibration and alarm threshold (G5) before any held-out read.

Fills configs/model_freeze.template.yaml into configs/model_freeze.yaml, sets only the
``threshold`` line of configs/alarm_policy.yaml, and appends a protocol_change record
to the research log. Refuses if the freeze record exists or the threshold is already set.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.research_log import append_record
from src.dataset_inventory import publish_new_bytes, sha256_file

THRESHOLD_LINE = re.compile(r"^threshold:[ \t]*null[ \t]*$", re.MULTILINE)
ANY_THRESHOLD = re.compile(r"^threshold:", re.MULTILINE)


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def git_commit(root: Path) -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if head.returncode != 0:
        raise SystemExit("cannot read git commit for the evaluation code")
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return head.stdout.strip(), bool(status.stdout.strip())


def load_yaml(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"{path}: root must be a mapping")
    return document


def find_manifests(record: dict) -> list[dict]:
    """Best-effort extraction of {manifest, sha256} entries from training_record.json."""
    for key in ("dataset_manifests", "train_datasets", "datasets", "training_datasets"):
        value = record.get(key)
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            entries = []
            for item in value:
                manifest = item.get("manifest") or item.get("extraction_manifest") or item.get("path")
                digest = item.get("manifest_sha256") or item.get("sha256")
                if manifest and digest:
                    entries.append({"manifest": str(manifest), "sha256": str(digest)})
            if entries:
                return entries
    return []


def replace_threshold(alarm_text: str, threshold: float) -> str:
    if not ANY_THRESHOLD.search(alarm_text):
        raise SystemExit("alarm_policy.yaml has no threshold key")
    if not THRESHOLD_LINE.search(alarm_text):
        raise SystemExit("alarm_policy.yaml threshold is already frozen (non-null); refusing")
    if len(THRESHOLD_LINE.findall(alarm_text)) != 1:
        raise SystemExit("alarm_policy.yaml must contain exactly one top-level threshold line")
    return THRESHOLD_LINE.sub(f"threshold: {threshold!r}", alarm_text, count=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--threshold-record", type=Path, required=True,
                        help="output of scripts/select_alarm_threshold.py on the calibrated table")
    parser.add_argument("--validation-predictions", type=Path, required=True,
                        help="immutable calibrated validation table the threshold was selected on")
    parser.add_argument("--frozen-by", required=True)
    parser.add_argument("--dataset-manifest", type=Path, action="append", default=[],
                        help="training extraction manifest(s); default read from training_record.json")
    parser.add_argument("--split-manifest", type=Path,
                        default=ROOT / "data/manifests/splits.template.yaml")
    parser.add_argument("--template", type=Path, default=ROOT / "configs/model_freeze.template.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/model_freeze.yaml")
    parser.add_argument("--alarm", type=Path, default=ROOT / "configs/alarm_policy.yaml")
    parser.add_argument("--feature-schema", type=Path, default=ROOT / "configs/feature_schema.yaml")
    parser.add_argument("--leakage-denylist", type=Path, default=ROOT / "configs/leakage_denylist.yaml")
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "RESEARCH PROTOCOL - AUTONOMOUS ROBOT RELIABILITY.md")
    parser.add_argument("--log", type=Path, default=ROOT / "logs/research-log.jsonl")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root for git and paths")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing freeze record {args.output}")
    alarm_text = args.alarm.read_text(encoding="utf-8")
    alarm = load_yaml(args.alarm)
    if alarm.get("threshold") is not None:
        raise SystemExit("alarm_policy.yaml threshold is already frozen (non-null); refusing")

    record_path = args.model_dir / "training_record.json"
    checkpoint = args.model_dir / "checkpoint.pt"
    normalization = args.model_dir / "normalization.json"
    for path in (record_path, checkpoint, normalization, args.calibration_report,
                 args.threshold_record, args.validation_predictions, args.template,
                 args.split_manifest, args.feature_schema, args.leakage_denylist, args.protocol):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    model_id = record.get("model_id")
    if not model_id:
        raise SystemExit("training_record.json lacks model_id")
    checkpoint_sha = sha256_file(checkpoint)
    declared = args.model_dir / "checkpoint.sha256"
    if declared.exists():
        stated = declared.read_text(encoding="utf-8").split()[0]
        if stated != checkpoint_sha:
            raise SystemExit("checkpoint.sha256 does not match checkpoint.pt")
    config_sha = record.get("config_sha256") or record.get("training_config_sha256")
    if not config_sha:
        raise SystemExit("training_record.json lacks config_sha256")

    calibration = load_yaml(args.calibration_report)
    if calibration.get("model_id") != model_id:
        raise SystemExit(
            f"calibration report model_id {calibration.get('model_id')} != {model_id}"
        )
    if calibration.get("selection_split") != "validation" or calibration.get("protected_test_used") is not False:
        raise SystemExit("calibration report must be validation-only with protected_test_used false")
    calibrator_path = Path(str(calibration["calibrator_artifact"]))
    if not calibrator_path.is_absolute():
        calibrator_path = args.root / calibrator_path
    if not calibrator_path.exists():
        raise SystemExit(f"calibrator artifact missing: {calibrator_path}")
    calibrator_sha = sha256_file(calibrator_path)
    if calibrator_sha != calibration.get("calibrator_artifact_sha256"):
        raise SystemExit("calibrator artifact sha256 does not match the calibration report")

    threshold_record = load_yaml(args.threshold_record)
    if threshold_record.get("selection_split") != "validation" or threshold_record.get("protected_test_used") is not False:
        raise SystemExit("threshold record must be validation-only with protected_test_used false")
    threshold = threshold_record.get("threshold")
    if threshold is None or not 0.0 <= float(threshold) <= 1.0:
        raise SystemExit("threshold record lacks a valid threshold")
    budget = float(alarm["false_alert_budget_per_clean_mission"])
    if float(threshold_record.get("false_alert_budget_per_clean_mission", budget)) != budget:
        raise SystemExit("threshold record budget differs from alarm_policy.yaml")

    manifests = [
        {"manifest": relative(path, args.root), "sha256": sha256_file(path)}
        for path in args.dataset_manifest
    ] or find_manifests(record)
    if not manifests:
        raise SystemExit("no training dataset manifest: pass --dataset-manifest or record it in training_record.json")
    new_alarm_text = replace_threshold(alarm_text, float(threshold))
    commit, dirty = git_commit(args.root)
    now = datetime.now(timezone.utc).isoformat()

    template = load_yaml(args.template)
    freeze = dict(template)
    freeze.update({"frozen": True, "protected_outcomes_consulted": False,
                   "frozen_utc": now, "frozen_by": args.frozen_by})
    freeze["dataset"] = {
        "manifest": manifests[0]["manifest"] if len(manifests) == 1 else [m["manifest"] for m in manifests],
        "manifest_sha256": manifests[0]["sha256"] if len(manifests) == 1 else [m["sha256"] for m in manifests],
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "feature_schema_sha256": sha256_file(args.feature_schema),
        "leakage_denylist_sha256": sha256_file(args.leakage_denylist),
    }
    freeze["predictor"] = {
        "model_id": model_id,
        "checkpoint": relative(checkpoint, args.root),
        "checkpoint_sha256": checkpoint_sha,
        "normalization_bundle_sha256": sha256_file(normalization),
        "training_config_sha256": str(config_sha),
    }
    freeze["calibration"] = {
        "calibration_id": str(calibration.get("calibration_id")),
        "artifact": relative(calibrator_path, args.root),
        "artifact_sha256": calibrator_sha,
    }
    freeze["alarm_policy"] = {
        "config": relative(args.alarm, args.root),
        "config_sha256": hashlib.sha256(new_alarm_text.encode("utf-8")).hexdigest(),
        "validation_false_alarm_budget": budget,
        "threshold": float(threshold),
        "threshold_record": relative(args.threshold_record, args.root),
        "threshold_record_sha256": sha256_file(args.threshold_record),
    }
    freeze["analysis"] = {
        "analysis_plan": relative(args.protocol, args.root),
        "analysis_plan_sha256": sha256_file(args.protocol),
        "evaluation_code_commit": commit,
        "evaluation_code_tree_dirty": dirty,
        "immutable_validation_predictions": relative(args.validation_predictions, args.root),
        "immutable_validation_predictions_sha256": sha256_file(args.validation_predictions),
        "calibration_report": relative(args.calibration_report, args.root),
        "calibration_report_sha256": sha256_file(args.calibration_report),
    }
    freeze["declaration"] = {
        "model_selection_complete": True,
        "calibration_selection_complete": True,
        "threshold_selection_complete": True,
        "protected_maps_or_outcomes_inspected": False,
    }
    freeze_text = yaml.safe_dump(freeze, sort_keys=False, allow_unicode=True)
    leftovers = [line for line in freeze_text.splitlines() if "TODO" in line]
    if leftovers:
        raise SystemExit(f"freeze record still contains TODO fields: {leftovers}")

    log_metadata = {
        "stage": "G5_predictor_freeze", "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha, "calibrator_sha256": calibrator_sha,
        "calibration_method": calibration.get("chosen_method"),
        "threshold": float(threshold), "evaluation_code_commit": commit,
        "freeze_record": relative(args.output, args.root), "protected_test_used": False,
    }
    if args.dry_run:
        print("DRY RUN: would write", args.output)
        print(freeze_text)
        print("DRY RUN: would set alarm_policy.yaml line ->", f"threshold: {float(threshold)!r}")
        print("DRY RUN: would append protocol_change record:", json.dumps(log_metadata, sort_keys=True))
        return 0

    publish_new_bytes(args.output, freeze_text.encode("utf-8"))
    args.alarm.write_text(new_alarm_text, encoding="utf-8")
    append_record(
        args.log, kind="protocol_change", actor=args.frozen_by,
        message=(
            f"G5 predictor freeze: {model_id} with {calibration.get('chosen_method')} calibration "
            f"and validation threshold {float(threshold)!r}; held-out maps remain unread"
        ),
        metadata=log_metadata,
    )
    print(f"froze {model_id}: {args.output}; alarm threshold={float(threshold)!r}; log appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

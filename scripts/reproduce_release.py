#!/usr/bin/env python3
"""One-command clean reproduction (``make reproduce``).

In a fresh detached git worktree of HEAD: run the test suite, the three inventory
validators and the raw-payload audit; regenerate predictions from the frozen
checkpoint on the validation and held-out derived data; recompute the results
tables; and diff them against the released tables (byte equality first, then a
numeric tolerance). Writes the immutable ``reports/reproduction/independent_rerun.yaml``
in the main repository. ``--dry-run`` prints the plan and touches nothing.

Raw and derived data, and model checkpoints, are content-addressed and shared with
the worktree through read-only symlinks; their checksums are verified against the
freeze record before use.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_inventory import publish_new_bytes  # noqa: E402
from src.release import git_head, sha256_file  # noqa: E402


SHARED_INPUTS = ("data/raw", "data/derived", "models", "reports/unseen_family", "reports/predictions/ablations")
# Released alarmed held-out tables of the comparison models; the rerun regenerates the
# frozen primary predictor (P3) and reuses these read-only (they are not regenerated).
SECONDARY_HELD_OUT_TABLES = ("p1_threshold_rules.alarmed.csv", "p4_gru.alarmed.csv", "p5_compact_transformer.alarmed.csv")
RELEASED_HELD_OUT_DIR = "reports/predictions/held_out/held_out_map_v1"
TABLE_NAMES = ("tab01_predictor_summary", "tab02_recall_by_family", "tab03_unseen_family", "tab04_ablations",
               "tab05_recovery_outcomes", "tab06_action_confusion", "tab07_latency")


def plan(root: Path, worktree: Path, *, venv_python: str, validation_dataset: str,
         held_out_dataset: str | None, model_dir: str | None, calibrator: str | None = None) -> list[dict[str, Any]]:
    py = sys.executable
    predictions = worktree / "reports/predictions"
    steps: list[dict[str, Any]] = [
        {"id": "tests", "command": [py, "-m", "pytest", "-q"], "cwd": str(worktree)},
        {"id": "development_inventory", "command": [py, "scripts/validate_development_dataset_inventory.py"], "cwd": str(worktree)},
        {"id": "validation_inventory", "command": [py, "scripts/validate_validation_dataset_inventory.py"], "cwd": str(worktree)},
        {"id": "targeted_inventory", "command": [py, "scripts/validate_targeted_dataset_inventory.py"], "cwd": str(worktree)},
        {"id": "raw_payload_audit", "command": [py, "scripts/audit_raw_artifact_payloads.py"], "cwd": str(worktree)},
    ]
    model = model_dir or "<configs/model_freeze.yaml predictor.checkpoint directory>"
    held = held_out_dataset or "<held-out dataset id from reports/confirmatory/held_out_map.yaml>"
    calibrator_path = calibrator or "<configs/model_freeze.yaml calibration.artifact>"
    for name, dataset, extra in (("validation", validation_dataset, []),
                                 ("held_out_map", held, ["--allow-protected-after-freeze"])):
        raw = predictions / f"rerun_{name}.csv"
        calibrated = predictions / f"rerun_{name}_calibrated.csv"
        alarmed = predictions / f"rerun_{name}_alarmed.csv"
        steps.extend([
            {"id": f"predict_{name}", "command": [venv_python, "scripts/predict_decisions.py", "--model-dir", model,
                                                  "--dataset", dataset, "--output", str(raw), *extra], "cwd": str(worktree)},
            {"id": f"calibrate_{name}", "command": [py, "scripts/apply_calibration.py", str(raw), calibrator_path, str(calibrated), *extra], "cwd": str(worktree)},
            {"id": f"alarm_{name}", "command": [py, "scripts/apply_alarm_policy.py", str(calibrated), str(alarmed), *extra], "cwd": str(worktree)},
        ])
    steps.append({
        "id": "tables", "command": [
            py, "scripts/build_tables.py", "--root", str(worktree), "--output-dir", str(worktree / "reports/tables_rerun"),
            "--input", f"predictions_validation={predictions / 'rerun_validation_alarmed.csv'}",
            "--input", f"predictions_held_out={predictions / 'rerun_held_out'}",
        ], "cwd": str(worktree),
    })
    return steps


def stage_held_out_tables(root: Path, worktree: Path) -> dict[str, str]:
    """reports/predictions/rerun_held_out/: the regenerated P3 table (created by the alarm
    step) beside read-only links to the released comparison-model tables."""
    staging = worktree / "reports/predictions/rerun_held_out"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "p3_causal_tcn.alarmed.csv").symlink_to(worktree / "reports/predictions/rerun_held_out_map_alarmed.csv")
    shared: dict[str, str] = {}
    for name in SECONDARY_HELD_OUT_TABLES:
        source = root / RELEASED_HELD_OUT_DIR / name
        if source.is_file():
            (staging / name).symlink_to(source.resolve())
            shared[name] = sha256_file(source)
    return shared


def numeric_diff(released: Path, rerun: Path, tolerance: float) -> dict[str, Any]:
    if sha256_file(released) == sha256_file(rerun):
        return {"status": "identical_bytes"}
    with released.open(newline="", encoding="utf-8") as a, rerun.open(newline="", encoding="utf-8") as b:
        rows_a, rows_b = list(csv.DictReader(a)), list(csv.DictReader(b))
    if len(rows_a) != len(rows_b) or (rows_a and rows_a[0].keys() != rows_b[0].keys()):
        return {"status": "shape_mismatch", "released_rows": len(rows_a), "rerun_rows": len(rows_b)}
    worst = 0.0
    mismatches = 0
    for left, right in zip(rows_a, rows_b):
        for key in left:
            x, y = left[key], right[key]
            if x == y:
                continue
            try:
                fx, fy = float(x), float(y)
            except ValueError:
                mismatches += 1
                continue
            if math.isnan(fx) and math.isnan(fy):
                continue
            gap = abs(fx - fy) / max(1.0, abs(fx))
            worst = max(worst, gap)
            if gap > tolerance:
                mismatches += 1
    return {"status": "within_tolerance" if mismatches == 0 else "differs",
            "max_relative_difference": worst, "mismatched_cells": mismatches, "tolerance": tolerance}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--venv-python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--validation-dataset", default="balanced_validation_v1-validation-324")
    parser.add_argument("--held-out-dataset", default=None)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--keep-worktree", action="store_true")
    parser.add_argument("--worktree-dir", type=Path, default=Path(tempfile.gettempdir()),
                        help="parent directory for the detached clean worktree")
    args = parser.parse_args()
    root = args.root
    output = args.output or (root / "reports/reproduction/independent_rerun.yaml")
    if output.exists():
        raise SystemExit(f"refusing to overwrite immutable rerun record: {output}")
    freeze_path = root / "configs/model_freeze.yaml"
    freeze = yaml.safe_load(freeze_path.read_text(encoding="utf-8")) if freeze_path.exists() else {}
    model_dir = None
    if freeze.get("frozen") is True:
        checkpoint = str(freeze.get("predictor", {}).get("checkpoint", ""))
        model_dir = str(Path(checkpoint).parent) if checkpoint else None
    held_out = args.held_out_dataset
    confirmatory = root / "reports/confirmatory/held_out_map.yaml"
    if held_out is None and confirmatory.exists():
        held_out = (yaml.safe_load(confirmatory.read_text(encoding="utf-8")) or {}).get("dataset_id")
    worktree = args.worktree_dir / f"research2-rerun-{int(time.time())}"
    calibrator = str(freeze.get("calibration", {}).get("artifact")) if freeze.get("frozen") is True else None
    steps = plan(root, worktree, venv_python=args.venv_python, validation_dataset=args.validation_dataset,
                 held_out_dataset=held_out, model_dir=model_dir, calibrator=calibrator)
    if args.dry_run:
        print(json.dumps({
            "worktree": str(worktree), "git_head": git_head(root), "shared_inputs_symlinked": list(SHARED_INPUTS),
            "steps": [{"id": step["id"], "command": " ".join(step["command"])} for step in steps],
            "diff": {"released": "reports/tables/<table>.csv", "rerun": "reports/tables_rerun/<table>.csv",
                     "rule": f"byte equality, else relative numeric tolerance {args.tolerance}"},
            "output": str(output),
        }, indent=2))
        return 0
    if freeze.get("frozen") is not True or model_dir is None:
        raise SystemExit("reproduction requires a frozen model record in configs/model_freeze.yaml")
    if held_out is None:
        raise SystemExit("held-out dataset id unknown; pass --held-out-dataset")
    checkpoint_path = root / freeze["predictor"]["checkpoint"]
    if sha256_file(checkpoint_path) != freeze["predictor"].get("checkpoint_sha256"):
        raise SystemExit("frozen checkpoint checksum mismatch; refusing to reproduce from a changed model")
    subprocess.run(["git", "worktree", "add", "--detach", str(worktree), "HEAD"], cwd=root, check=True)
    record: dict[str, Any] = {
        "schema_version": 1, "evidence_type": "independent_clean_rerun", "protected_test_used": True,
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_head(root), "worktree": str(worktree),
        "frozen_checkpoint_sha256": freeze["predictor"].get("checkpoint_sha256"), "steps": [], "diffs": {},
    }
    try:
        for name in SHARED_INPUTS:
            source = root / name
            if source.exists():
                (worktree / name).parent.mkdir(parents=True, exist_ok=True)
                target = worktree / name
                if target.is_symlink() or target.exists():
                    shutil.rmtree(target) if target.is_dir() and not target.is_symlink() else target.unlink()
                target.symlink_to(source.resolve())
        (worktree / "reports/predictions").mkdir(parents=True, exist_ok=True)
        record["released_secondary_tables_reused_sha256"] = stage_held_out_tables(root, worktree)
        ok = True
        for step in steps:
            if not ok:
                record["steps"].append({"id": step["id"], "status": "skipped"})
                continue
            started = time.monotonic()
            result = subprocess.run(step["command"], cwd=step["cwd"], capture_output=True, text=True, check=False)
            record["steps"].append({
                "id": step["id"], "command": " ".join(step["command"]), "returncode": result.returncode,
                "seconds": round(time.monotonic() - started, 1),
                "stdout_tail": result.stdout[-2000:], "stderr_tail": result.stderr[-2000:],
            })
            ok = ok and result.returncode == 0
        if ok:
            for table in TABLE_NAMES:
                released = root / "reports/tables" / f"{table}.csv"
                rerun = worktree / "reports/tables_rerun" / f"{table}.csv"
                if released.exists() and rerun.exists():
                    record["diffs"][table] = numeric_diff(released, rerun, args.tolerance)
                else:
                    record["diffs"][table] = {"status": "missing", "released": released.exists(), "rerun": rerun.exists()}
            for name in ("rerun_validation_alarmed.csv", "rerun_held_out_map_alarmed.csv"):
                path = worktree / "reports/predictions" / name
                if path.exists():
                    record.setdefault("rerun_prediction_sha256", {})[name] = sha256_file(path)
        record["passed"] = bool(ok and record["diffs"] and all(
            item["status"] in {"identical_bytes", "within_tolerance"} for item in record["diffs"].values()))
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not args.keep_worktree:
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=root, check=False)
    publish_new_bytes(output, yaml.safe_dump(record, sort_keys=False).encode("utf-8"))
    print(f"wrote {output}: passed={record['passed']}")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

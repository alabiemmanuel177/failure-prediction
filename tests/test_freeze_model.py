import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from confirmatory_fixtures import write_unfrozen_alarm_policy
from scripts.research_log import read_records, verify
from src.evaluation.calibrators import Calibrator
from synthetic_prediction_tables import build_rows, write_table

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def freeze_env(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    shutil.copy(ROOT / "configs/model_freeze.template.yaml", configs / "model_freeze.template.yaml")
    # The workspace policy may already carry the frozen threshold; the fixture always
    # starts from the pre-freeze ``threshold: null`` line.
    write_unfrozen_alarm_policy(configs / "alarm_policy.yaml")
    for name in ("feature_schema.yaml", "leakage_denylist.yaml"):
        (configs / name).write_text(f"schema_version: 1\nname: {name}\n")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("# protocol copy\n")
    splits = tmp_path / "splits.yaml"
    splits.write_text("development: {maps: [dev_00]}\n")
    manifest = tmp_path / "extraction_manifest.jsonl"
    manifest.write_text('{"run_id": "x", "split": "development"}\n')

    model_dir = tmp_path / "models" / "p3_run"
    model_dir.mkdir(parents=True)
    (model_dir / "checkpoint.pt").write_bytes(b"fake-checkpoint-bytes")
    (model_dir / "checkpoint.sha256").write_text(sha(model_dir / "checkpoint.pt") + "\n")
    (model_dir / "normalization.json").write_text('{"mean": [0.0]}')
    (model_dir / "training_record.json").write_text(json.dumps({
        "model_id": "p3_causal_tcn", "config_sha256": "c" * 64, "seed": 20260903,
        "dataset_manifests": [{"manifest": str(manifest), "sha256": sha(manifest)}],
    }))

    calibrator = tmp_path / "p3.calibrator.json"
    calibrator.write_bytes(Calibrator("temperature_scaling", {"temperature": 1.3}).to_json_bytes())
    calibration_report = tmp_path / "p3.calibration.yaml"
    calibration_report.write_text(yaml.safe_dump({
        "model_id": "p3_causal_tcn", "selection_split": "validation", "protected_test_used": False,
        "calibration_id": "p3_causal_tcn:temperature_scaling:abc", "chosen_method": "temperature_scaling",
        "calibrator_artifact": str(calibrator), "calibrator_artifact_sha256": sha(calibrator),
    }))
    predictions = write_table(tmp_path / "p3.validation.csv", build_rows("p3_causal_tcn"))
    threshold_record = tmp_path / "threshold.yaml"
    threshold_record.write_text(yaml.safe_dump({
        "selection_split": "validation", "protected_test_used": False, "threshold": 0.9,
        "false_alert_budget_per_clean_mission": 0.10,
    }))
    log = tmp_path / "research-log.jsonl"
    return {
        "tmp": tmp_path, "configs": configs, "model_dir": model_dir, "protocol": protocol,
        "splits": splits, "calibration_report": calibration_report, "calibrator": calibrator,
        "predictions": predictions, "threshold_record": threshold_record, "log": log,
    }


def freeze_command(env, *extra):
    configs = env["configs"]
    return [
        sys.executable, str(ROOT / "scripts/freeze_model.py"),
        "--model-dir", str(env["model_dir"]),
        "--calibration-report", str(env["calibration_report"]),
        "--threshold-record", str(env["threshold_record"]),
        "--validation-predictions", str(env["predictions"]),
        "--frozen-by", "Test Researcher",
        "--template", str(configs / "model_freeze.template.yaml"),
        "--output", str(configs / "model_freeze.yaml"),
        "--alarm", str(configs / "alarm_policy.yaml"),
        "--feature-schema", str(configs / "feature_schema.yaml"),
        "--leakage-denylist", str(configs / "leakage_denylist.yaml"),
        "--protocol", str(env["protocol"]),
        "--split-manifest", str(env["splits"]),
        "--log", str(env["log"]),
        *extra,
    ]


def test_dry_run_changes_nothing(freeze_env):
    alarm = freeze_env["configs"] / "alarm_policy.yaml"
    before = alarm.read_bytes()
    result = subprocess.run(freeze_command(freeze_env, "--dry-run"), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout and "threshold: 0.9" in result.stdout
    assert not (freeze_env["configs"] / "model_freeze.yaml").exists()
    assert alarm.read_bytes() == before
    assert not freeze_env["log"].exists()


def test_freeze_fills_template_sets_only_threshold_and_logs(freeze_env):
    alarm = freeze_env["configs"] / "alarm_policy.yaml"
    before_lines = alarm.read_text().splitlines(keepends=True)
    result = subprocess.run(freeze_command(freeze_env), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    freeze = yaml.safe_load((freeze_env["configs"] / "model_freeze.yaml").read_text())
    assert freeze["frozen"] is True
    assert freeze["protected_outcomes_consulted"] is False
    assert freeze["frozen_by"] == "Test Researcher"
    assert "TODO" not in (freeze_env["configs"] / "model_freeze.yaml").read_text()
    assert freeze["predictor"]["model_id"] == "p3_causal_tcn"
    assert freeze["predictor"]["checkpoint_sha256"] == sha(freeze_env["model_dir"] / "checkpoint.pt")
    assert freeze["predictor"]["normalization_bundle_sha256"] == sha(freeze_env["model_dir"] / "normalization.json")
    assert freeze["predictor"]["training_config_sha256"] == "c" * 64
    assert freeze["calibration"]["artifact_sha256"] == sha(freeze_env["calibrator"])
    assert freeze["dataset"]["manifest_sha256"] == sha(freeze_env["tmp"] / "extraction_manifest.jsonl")
    assert freeze["dataset"]["split_manifest_sha256"] == sha(freeze_env["splits"])
    assert freeze["analysis"]["analysis_plan_sha256"] == sha(freeze_env["protocol"])
    assert freeze["analysis"]["immutable_validation_predictions_sha256"] == sha(freeze_env["predictions"])
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert freeze["analysis"]["evaluation_code_commit"] == commit
    assert freeze["alarm_policy"]["config_sha256"] == sha(alarm)
    assert freeze["alarm_policy"]["threshold"] == 0.9
    assert freeze["declaration"] == {
        "model_selection_complete": True, "calibration_selection_complete": True,
        "threshold_selection_complete": True, "protected_maps_or_outcomes_inspected": False,
    }

    after_lines = alarm.read_text().splitlines(keepends=True)
    assert len(before_lines) == len(after_lines)
    changed = [(a, b) for a, b in zip(before_lines, after_lines) if a != b]
    assert changed == [("threshold: null\n", "threshold: 0.9\n")]
    assert yaml.safe_load(alarm.read_text())["threshold"] == 0.9

    records = read_records(freeze_env["log"])
    verify(records)
    assert len(records) == 1
    assert records[0]["kind"] == "protocol_change"
    assert records[0]["metadata"]["threshold"] == 0.9
    assert records[0]["metadata"]["protected_test_used"] is False


def test_freeze_refuses_to_overwrite_or_refreeze_threshold(freeze_env):
    first = subprocess.run(freeze_command(freeze_env), capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    freeze_bytes = (freeze_env["configs"] / "model_freeze.yaml").read_bytes()
    alarm_bytes = (freeze_env["configs"] / "alarm_policy.yaml").read_bytes()
    second = subprocess.run(freeze_command(freeze_env), capture_output=True, text=True)
    assert second.returncode != 0 and "refusing to overwrite" in second.stderr
    assert (freeze_env["configs"] / "model_freeze.yaml").read_bytes() == freeze_bytes
    assert (freeze_env["configs"] / "alarm_policy.yaml").read_bytes() == alarm_bytes
    assert len(read_records(freeze_env["log"])) == 1

    # A fresh output path must still refuse while alarm_policy.yaml already carries a threshold.
    third = subprocess.run(
        freeze_command(freeze_env, "--output", str(freeze_env["tmp"] / "other_freeze.yaml")),
        capture_output=True, text=True,
    )
    assert third.returncode != 0 and "already frozen" in third.stderr
    assert not (freeze_env["tmp"] / "other_freeze.yaml").exists()


def test_freeze_refuses_inconsistent_inputs(freeze_env):
    (freeze_env["model_dir"] / "checkpoint.sha256").write_text("0" * 64 + "\n")
    result = subprocess.run(freeze_command(freeze_env), capture_output=True, text=True)
    assert result.returncode != 0 and "checkpoint.sha256" in result.stderr
    assert not (freeze_env["configs"] / "model_freeze.yaml").exists()

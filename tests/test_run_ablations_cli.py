from pathlib import Path
import subprocess
import sys

import yaml

from src.evaluation.calibrators import Calibrator
from synthetic_prediction_tables import build_rows, write_table

ROOT = Path(__file__).resolve().parents[1]

ABLATIONS = {
    "ablations": {
        "no_localisation": {"kind": "feature_group", "group": "localisation"},
        "single_timestamp": {"kind": "input_policy"},
        "no_calibration": {"kind": "policy"},
        "no_persistence": {"kind": "policy", "persistence": {"required_above_threshold": 1, "decisions_considered": 1}},
    }
}


def build_env(tmp_path):
    calibrator = tmp_path / "calibrator.json"
    calibrator.write_bytes(Calibrator("identity").to_json_bytes())
    alarm = tmp_path / "alarm_policy.yaml"
    document = yaml.safe_load((ROOT / "configs/alarm_policy.yaml").read_text())
    document["threshold"] = 0.9
    alarm.write_text(yaml.safe_dump(document, sort_keys=False))
    freeze = tmp_path / "model_freeze.yaml"
    freeze.write_text(yaml.safe_dump({
        "frozen": True, "protected_outcomes_consulted": False,
        "predictor": {"model_id": "p3_causal_tcn"},
        "calibration": {"artifact": str(calibrator),
                        "artifact_sha256": Calibrator("identity").sha256()},
        "alarm_policy": {"threshold": 0.9},
    }))
    ablations = tmp_path / "ablations.yaml"
    ablations.write_text(yaml.safe_dump(ABLATIONS))
    predictions = write_table(tmp_path / "p3.calibrated.csv", build_rows("p3_causal_tcn"))
    config = tmp_path / "p3.yaml"
    config.write_text("model: p3\n")
    return {"alarm": alarm, "freeze": freeze, "ablations": ablations, "predictions": predictions, "config": config}


def command(env, tmp_path, *extra):
    return [
        sys.executable, str(ROOT / "scripts/run_ablations.py"),
        "--freeze", str(env["freeze"]), "--alarm", str(env["alarm"]), "--ablations", str(env["ablations"]),
        "--primary-predictions", str(env["predictions"]), "--config", str(env["config"]),
        "--train-dataset", "balanced_pilot_v1-development-648",
        "--selection-dataset", "balanced_validation_v1-validation-324",
        "--output-root", str(tmp_path / "models"), "--predictions-root", str(tmp_path / "predictions"),
        "--output", str(tmp_path / "validation_ablations.yaml"), "--python", "PYTHON",
        *extra,
    ]


def test_dry_run_prints_commands_and_writes_nothing(tmp_path):
    env = build_env(tmp_path)
    result = subprocess.run(command(env, tmp_path, "--dry-run"), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--ablation no_localisation" in result.stdout
    assert "--ablation single_timestamp" in result.stdout
    assert "train_predictor.py" in result.stdout and "predict_decisions.py" in result.stdout
    assert "nice -n 19" in result.stdout
    assert result.stdout.count("--ablation") == 2  # policy ablations never retrain
    assert not (tmp_path / "validation_ablations.yaml").exists()
    assert not (tmp_path / "models").exists()


def test_policy_only_ablations_evaluate_without_training(tmp_path):
    env = build_env(tmp_path)
    result = subprocess.run(command(env, tmp_path, "--only", "no_calibration", "--only", "no_persistence"),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = yaml.safe_load((tmp_path / "validation_ablations.yaml").read_text())
    assert report["confirmatory"] is False and report["protected_test_used"] is False
    assert report["primary"]["result"]["event_recall"] == 0.75
    persistence = report["ablations"]["no_persistence"]
    assert persistence["retrained"] is False
    assert persistence["result"]["policy"]["persistence"] == "1-of-1"
    assert persistence["result"]["lead_time"]["median_seconds_detected"] == 5.0
    assert persistence["delta_vs_primary"]["point_estimate"] == 0.0
    assert report["ablations"]["no_calibration"]["calibration_mode"] == "none"


def test_refuses_without_freeze(tmp_path):
    env = build_env(tmp_path)
    env["freeze"] = tmp_path / "absent.yaml"
    result = subprocess.run(command(env, tmp_path, "--dry-run"), capture_output=True, text=True)
    assert result.returncode != 0 and "freeze missing" in result.stderr

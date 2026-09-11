"""End-to-end smoke of train -> predict -> latency on synthetic contract-layout data."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from models_synthetic import build_dataset

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
COLUMNS = [
    "run_id", "decision_index", "decision_time", "split", "map_id", "route_id", "fault_family",
    "severity", "seed", "protected_test_used", "eligibility", "label", "primary_event_class",
    "primary_event_time", "model_id", "raw_score", "risk_score",
]


def run(*arguments, expect=0):
    result = subprocess.run([PYTHON, *map(str, arguments)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == expect, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    root = tmp_path_factory.mktemp("derived")
    build_dataset(root, "syn_dev-development-8", "development", episodes=8, seed=1, decisions=70)
    build_dataset(root, "syn_val-validation-6", "validation", episodes=6, seed=2, decisions=70)
    return root


def train(datasets, models_root, model_id, *extra, expect=0):
    return run(
        ROOT / "scripts/train_predictor.py", "--model-id", model_id,
        "--config", ROOT / "configs/models" / f"{model_id}.yaml",
        "--train-dataset", "syn_dev-development-8", "--selection-dataset", "syn_val-validation-6",
        "--seed", 3, "--output-root", models_root, "--derived-root", datasets,
        "--max-epochs", 2, "--device", "cpu", *extra, expect=expect,
    )


def test_tcn_train_predict_and_latency_follow_the_contract(datasets, tmp_path):
    models_root = tmp_path / "models"
    train(datasets, models_root, "p3_causal_tcn", "--ablation", "no_localisation")
    model_dir = models_root / "p3_causal_tcn__abl-no_localisation__seed3"
    for name in ("checkpoint.pt", "normalization.json", "training_record.json", "checkpoint.sha256"):
        assert (model_dir / name).is_file(), name
    record = json.loads((model_dir / "training_record.json").read_text())
    for key in ("seed", "config_sha256", "train_datasets", "selection_dataset", "development",
                "validation", "fitting_windows", "epochs_run", "early_stopping", "parameter_count",
                "wall_seconds", "gpu_name", "git_commit", "history", "checkpoint_sha256"):
        assert key in record, key
    assert record["early_stopping"]["metric"] == "validation_auprc"
    assert record["protected_test_used"] is False
    assert record["fitting_windows"]["natural_prevalence"] != record["fitting_windows"]["rebalanced_prevalence"]
    assert record["feature_mask"]["masked_features"] == ["pose_covariance_trace", "pose_jump"]
    assert record["event_recall_log_basis"] == "decisions"
    assert record["history"][0]["event_recall_at_frozen_budget"]["purpose"] == "information_only"
    bundle = json.loads((model_dir / "normalization.json").read_text())
    assert bundle["fit_split"] == "development" and "pose_jump__age_seconds" in bundle["means"]
    # never overwrite a run directory
    train(datasets, models_root, "p3_causal_tcn", "--ablation", "no_localisation", expect=1)

    output = tmp_path / "predictions" / "val.csv"
    run(ROOT / "scripts/predict_decisions.py", "--model-dir", model_dir,
        "--dataset", "syn_val-validation-6", "--output", output, "--derived-root", datasets)
    rows = list(csv.DictReader(output.open(newline="")))
    assert list(rows[0].keys()) == COLUMNS
    assert len(rows) == 6 * 70
    keys = [(r["run_id"], int(r["decision_index"])) for r in rows]
    assert keys == sorted(keys)
    assert {r["split"] for r in rows} == {"validation"}
    assert {r["protected_test_used"] for r in rows} == {"false"}
    assert {r["label"] for r in rows} <= {"-1", "0", "1"}
    assert all(0.0 <= float(r["raw_score"]) <= 1.0 and r["raw_score"] == r["risk_score"] for r in rows)
    assert all((r["label"] == "-1") == r["eligibility"].startswith("excluded") for r in rows)
    assert (output.with_name("val.csv.provenance.json")).is_file()
    run(ROOT / "scripts/predict_decisions.py", "--model-dir", model_dir,
        "--dataset", "syn_val-validation-6", "--output", output, "--derived-root", datasets, expect=1)

    latency = run(ROOT / "scripts/measure_inference_latency.py", "--model-dir", model_dir,
                  "--decisions", 40, "--warmup", 5, "--output", tmp_path / "latency.json")
    report = json.loads((tmp_path / "latency.json").read_text())
    assert report["device"] == "cpu" and report["batch_size"] == 1
    assert report["end_to_end"]["count"] == 35
    assert all(report[part]["p95_ms"] >= report[part]["median_ms"] for part in
               ("feature_preparation", "model_forward", "alarm_policy", "end_to_end"))
    assert "median_ms" in json.loads(latency.stdout)["end_to_end"]


def test_training_refuses_wrong_split_and_unknown_ablation(datasets, tmp_path):
    models_root = tmp_path / "models"
    result = run(
        ROOT / "scripts/train_predictor.py", "--model-id", "p4_gru",
        "--config", ROOT / "configs/models/p4_gru.yaml",
        "--train-dataset", "syn_val-validation-6", "--selection-dataset", "syn_dev-development-8",
        "--seed", 3, "--output-root", models_root, "--derived-root", datasets, "--device", "cpu",
        expect=1,
    )
    assert "fitting" in result.stderr and "validation" in result.stderr
    result = train(datasets, models_root, "p4_gru", "--ablation", "bogus", expect=1)
    assert "unknown ablation" in result.stderr


def test_gru_transformer_and_autoencoder_train_with_family_exclusion(datasets, tmp_path):
    models_root = tmp_path / "models"
    train(datasets, models_root, "p4_gru", "--exclude-family", "wheel_slip")
    record = json.loads((models_root / "p4_gru__excl-wheel_slip__seed3/training_record.json").read_text())
    assert "wheel_slip" not in record["development"]["by_fault_family"]
    assert "wheel_slip" not in record["validation"]["by_fault_family"]
    train(datasets, models_root, "p5_compact_transformer")
    train(datasets, models_root, "p2_reconstruction_ae")
    ae_record = json.loads((models_root / "p2_reconstruction_ae__seed3/training_record.json").read_text())
    assert ae_record["fitting_windows"]["fitting_positive_windows"] == 0
    assert ae_record["fitting_windows"]["episodes_used"] == 2       # clean development episodes only
    assert ae_record["error_scaler"]["fit_split"] == "development"
    output = tmp_path / "ae.csv"
    run(ROOT / "scripts/predict_decisions.py", "--model-dir", models_root / "p2_reconstruction_ae__seed3",
        "--dataset", "syn_val-validation-6", "--output", output, "--derived-root", datasets)
    scores = [float(r["raw_score"]) for r in csv.DictReader(output.open(newline=""))]
    assert min(scores) >= 0.0 and max(scores) <= 1.0


def test_oracle_predictions_are_labels_and_marked_analysis_only(datasets, tmp_path):
    models_root = tmp_path / "models"
    run(ROOT / "scripts/train_predictor.py", "--model-id", "p6_oracle", "--seed", 1,
        "--output-root", models_root, "--derived-root", datasets)
    record = json.loads((models_root / "p6_oracle__seed1/training_record.json").read_text())
    assert record["analysis_only"] is True
    output = tmp_path / "oracle.csv"
    run(ROOT / "scripts/predict_decisions.py", "--model-dir", models_root / "p6_oracle__seed1",
        "--dataset", "syn_val-validation-6", "--output", output, "--derived-root", datasets)
    for row in csv.DictReader(output.open(newline="")):
        expected = 1.0 if row["eligibility"] == "eligible_positive" else 0.0
        assert float(row["risk_score"]) == expected


def test_prediction_refuses_protected_datasets_without_the_confirmatory_path(datasets, tmp_path):
    protected_root = tmp_path / "protected"
    build_dataset(protected_root, "syn_test-held_out_map_test-2", "held_out_map_test", episodes=2,
                  protected=True, map_prefix="hold")
    models_root = tmp_path / "models"
    run(ROOT / "scripts/train_predictor.py", "--model-id", "p6_oracle", "--seed", 1,
        "--output-root", models_root, "--derived-root", datasets)
    result = run(ROOT / "scripts/predict_decisions.py", "--model-dir", models_root / "p6_oracle__seed1",
                 "--dataset", "syn_test-held_out_map_test-2", "--output", tmp_path / "p.csv",
                 "--derived-root", protected_root, expect=1)
    assert "explicit approval" in result.stderr
    result = run(ROOT / "scripts/predict_decisions.py", "--model-dir", models_root / "p6_oracle__seed1",
                 "--dataset", "syn_test-held_out_map_test-2", "--output", tmp_path / "p.csv",
                 "--derived-root", protected_root, "--allow-protected-after-freeze", expect=1)
    # Either refusal is correct: before the freeze the confirmatory gate refuses; once the
    # repository holds a signed freeze the oracle's own contract refuses first.
    assert "confirmatory freeze" in result.stderr or "oracle must never score protected" in result.stderr
    assert not (tmp_path / "p.csv").exists()

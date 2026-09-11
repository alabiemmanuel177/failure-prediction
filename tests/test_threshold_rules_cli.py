"""P1 rule tuning and scoring CLIs on synthetic derived datasets (never real data)."""

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from models_synthetic import build_dataset
from scripts.tune_threshold_rules import main as tune_main
from scripts.predict_threshold_rules import main as predict_main

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = "synthetic_dev-development-8"
VALIDATION = "synthetic_val-validation-8"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def tune(tmp_path: Path, *extra: str) -> dict[str, Path]:
    derived = tmp_path / "derived"
    build_dataset(derived, DEVELOPMENT, "development", episodes=8, seed=1)
    build_dataset(derived, VALIDATION, "validation", episodes=8, seed=2)
    paths = {
        "derived": derived, "record": tmp_path / "fold/tuning_p1.yaml",
        "predictions": tmp_path / "fold/validation_p1.csv", "rules": tmp_path / "fold/rules_p1.yaml",
    }
    assert tune_main([
        "--development-dataset", DEVELOPMENT, "--validation-dataset", VALIDATION,
        "--derived-root", str(derived), "--output", str(paths["record"]),
        "--predictions", str(paths["predictions"]), "--rules-output", str(paths["rules"]),
        "--quantile-grid", "5", "--sweeps", "1", *extra,
    ]) == 0
    return paths


def test_exclude_family_drops_the_family_from_grid_and_tuning_and_freezes_fold_rules(tmp_path):
    baseline_bytes = (ROOT / "configs/baseline_rules.yaml").read_bytes()
    paths = tune(tmp_path, "--exclude-family", "wheel_slip")
    assert (ROOT / "configs/baseline_rules.yaml").read_bytes() == baseline_bytes
    record = yaml.safe_load(paths["record"].read_text(encoding="utf-8"))
    assert record["excluded_family"] == "wheel_slip"
    assert record["excluded_episode_counts"] == {"development": 2, "validation": 2}
    assert record["development_episode_count"] == record["validation_episode_count"] == 6
    assert record["protected_test_used"] is False and record["selection_split"] == "validation"
    assert record["rules_output"] == str(paths["rules"].resolve())
    predictions = read_csv_rows(paths["predictions"])
    assert predictions and "wheel_slip" not in {row["fault_family"] for row in predictions}
    assert {row["split"] for row in predictions} == {"validation"}
    assert {row["risk_score"] for row in predictions} <= {"0.0", "1.0"}

    rules = yaml.safe_load(paths["rules"].read_text(encoding="utf-8"))
    baseline = yaml.safe_load((ROOT / "configs/baseline_rules.yaml").read_text(encoding="utf-8"))
    assert rules["status"] == "frozen_after_validation_tuning"
    assert rules["excluded_family"] == "wheel_slip"
    assert set(rules["rules"]) == set(baseline["rules"])
    for name, spec in rules["rules"].items():
        assert spec["threshold"] == record["selected_thresholds"][name]
        assert {k: v for k, v in spec.items() if k != "threshold"} == {
            k: v for k, v in baseline["rules"][name].items() if k != "threshold"
        }
    assert rules["tuning_record"] == str(paths["record"].resolve())
    assert len(rules["tuning_record_sha256"]) == 64
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        tune_main(["--development-dataset", DEVELOPMENT, "--validation-dataset", VALIDATION,
                   "--derived-root", str(paths["derived"]), "--output", str(paths["record"]),
                   "--predictions", str(tmp_path / "other.csv"), "--rules-output", str(tmp_path / "r.yaml")])


def test_exclude_family_refuses_no_op_clean_and_config_writes(tmp_path):
    derived = tmp_path / "derived"
    build_dataset(derived, DEVELOPMENT, "development", episodes=8, seed=1)
    build_dataset(derived, VALIDATION, "validation", episodes=8, seed=2)
    base = ["--development-dataset", DEVELOPMENT, "--validation-dataset", VALIDATION,
            "--derived-root", str(derived), "--quantile-grid", "5", "--sweeps", "1"]
    with pytest.raises(SystemExit, match="matches no development episode"):
        tune_main([*base, "--exclude-family", "camera_occlusion", "--output", str(tmp_path / "a.yaml"),
                   "--predictions", str(tmp_path / "a.csv")])
    with pytest.raises(SystemExit, match="cannot be excluded"):
        tune_main([*base, "--exclude-family", "none", "--output", str(tmp_path / "b.yaml"),
                   "--predictions", str(tmp_path / "b.csv")])
    with pytest.raises(SystemExit, match="reserved for the full-data P1 freeze"):
        tune_main([*base, "--exclude-family", "wheel_slip", "--write-config",
                   "--output", str(tmp_path / "c.yaml"), "--predictions", str(tmp_path / "c.csv")])
    assert not (tmp_path / "a.yaml").exists() and not (tmp_path / "c.csv").exists()


def test_predict_scores_a_dataset_with_per_fold_rules(tmp_path):
    paths = tune(tmp_path, "--exclude-family", "wheel_slip")
    output = tmp_path / "held_out_p1.csv"
    assert predict_main(["--rules", str(paths["rules"]), "--dataset", VALIDATION,
                         "--derived-root", str(paths["derived"]), "--output", str(output)]) == 0
    rows = read_csv_rows(output)
    assert len(rows) == 8 * 60
    assert {row["model_id"] for row in rows} == {"p1_threshold_rules"}
    assert "wheel_slip" in {row["fault_family"] for row in rows}
    assert all(row["raw_score"] == row["risk_score"] in {"0.0", "1.0"} for row in rows)
    # The fold rules and the tuning predictions agree on the episodes the fold tuned on.
    tuned = {(row["run_id"], row["decision_index"]): row["risk_score"]
             for row in read_csv_rows(paths["predictions"])}
    scored = {(row["run_id"], row["decision_index"]): row["risk_score"] for row in rows}
    assert tuned and all(scored[key] == value for key, value in tuned.items())
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        predict_main(["--rules", str(paths["rules"]), "--dataset", VALIDATION,
                      "--derived-root", str(paths["derived"]), "--output", str(output)])
    unfrozen = tmp_path / "unfrozen.yaml"
    document = yaml.safe_load(paths["rules"].read_text(encoding="utf-8"))
    document["status"] = "draft"
    unfrozen.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(SystemExit, match="not frozen"):
        predict_main(["--rules", str(unfrozen), "--dataset", VALIDATION,
                      "--derived-root", str(paths["derived"]), "--output", str(tmp_path / "x.csv")])


def test_predict_refuses_protected_dataset_without_explicit_approval(tmp_path):
    paths = tune(tmp_path)
    protected = "synthetic_test-held_out_map_test-4"
    build_dataset(paths["derived"], protected, "held_out_map_test", episodes=4, seed=3,
                  protected=True, map_prefix="test")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/predict_threshold_rules.py"), "--rules", str(paths["rules"]),
         "--dataset", protected, "--derived-root", str(paths["derived"]),
         "--output", str(tmp_path / "protected.csv")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0 and "explicit approval" in result.stderr
    assert not (tmp_path / "protected.csv").exists()

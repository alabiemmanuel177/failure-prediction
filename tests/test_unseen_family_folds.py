from pathlib import Path

import pytest
import yaml

from confirmatory_fixtures import FAMILIES, ROOT, held_out_episodes, table_rows, write_table
from scripts.run_unseen_family_folds import evaluate_folds, fold_paths, main

BOOTSTRAP = {"replicates": 40, "seed": 5}


def fold_tables(work_root: Path, *, p3_detect: dict[str, float]) -> None:
    episodes = held_out_episodes(routes_per_map=2, seeds=2)
    for family in FAMILIES:
        subset = [item for item in episodes if item["fault_family"] in {family, "none"}]
        paths = fold_paths(work_root, family)
        write_table(paths["held_out_p3_alarmed"], table_rows(
            subset, model_id="p3_causal_tcn", protected=True,
            detect_probability=p3_detect[family], false_alert_probability=0.0, seed=1,
        ))
        write_table(paths["held_out_p1_alarmed"], table_rows(
            subset, model_id="p1_threshold_rules", protected=True,
            detect_probability=0.3, false_alert_probability=0.1, seed=2,
        ))
        for key in ("p3", "p1"):
            paths[f"threshold_{key}"].write_text(
                yaml.safe_dump({"threshold": 0.4 if key == "p3" else 0.6, "selection_split": "validation"}),
                encoding="utf-8",
            )


def test_dry_run_prints_fit_commands_per_family_without_protected_access(tmp_path, capsys):
    work = tmp_path / "work"
    assert main(["--stage", "fit", "--dry-run", "--work-root", str(work),
                 "--model-root", str(tmp_path / "models"),
                 "--train-dataset", "balanced_pilot_v1-development-648",
                 "--train-dataset", "targeted_development_v1-development-1212"]) == 0
    output = capsys.readouterr().out
    lines = output.splitlines()
    for family in FAMILIES:
        assert output.count(f"--exclude-family {family}") == 2  # P3 training and P1 tuning
        p1 = [line for line in lines if "tune_threshold_rules.py" in line and f"--exclude-family {family}" in line]
        assert len(p1) == 1
        paths = fold_paths(work, family)
        assert (f"--development-dataset balanced_pilot_v1-development-648 "
                f"--development-dataset targeted_development_v1-development-1212 "
                f"--validation-dataset balanced_validation_v1-validation-324") in p1[0]
        assert f"--output {paths['tuning_p1']}" in p1[0]
        assert f"--predictions {paths['validation_p1']}" in p1[0]
        assert f"--rules-output {paths['rules_p1']}" in p1[0]
        assert "--write-config" not in p1[0]
        assert f"[internal] write fixed P1 threshold 0.5 on the rule score -> {paths['threshold_p1']}" in lines
    # P3 keeps the contract training CLI; P1 never goes through train_predictor.py.
    assert output.count("train_predictor.py") == 7
    assert "p1_threshold_rules.yaml" not in output and "--model-id p1_threshold_rules" not in output
    assert output.count("select_calibration.py") == 7
    assert output.count("select_alarm_threshold.py") == 7  # P3 only; P1 uses the fixed 0.5
    assert "--allow-protected-after-freeze" not in output
    assert "configs/baseline_rules.yaml" not in output
    assert not work.exists()


def test_dry_run_evaluate_stage_requests_protected_predictions(tmp_path, capsys):
    work = tmp_path / "work"
    assert main(["--stage", "evaluate", "--dry-run", "--work-root", str(work),
                 "--model-root", str(tmp_path / "models"), "--output", str(tmp_path / "out.yaml")]) == 0
    output = capsys.readouterr().out
    lines = output.splitlines()
    # 7 families x (P3 predict + P1 predict + P3 calibration on held-out rows)
    assert output.count("--allow-protected-after-freeze") == 21
    assert output.count("predict_decisions.py") == 7
    for family in FAMILIES:
        paths = fold_paths(work, family)
        p1 = [line for line in lines if "predict_threshold_rules.py" in line and f"/{family}/" in line]
        assert len(p1) == 1
        assert f"--rules {paths['rules_p1']}" in p1[0]
        assert "--dataset held_out_map_v1-held_out_map_test-960" in p1[0]
        assert f"--output {paths['held_out_p1']}" in p1[0]
        assert p1[0].endswith("--allow-protected-after-freeze")
        assert f"apply fold threshold from {paths['threshold_p1']}" in output
    assert "predict_decisions.py --model-dir" in output and "p1_threshold_rules" not in output.replace(
        "predict_threshold_rules.py", "")
    assert not (tmp_path / "out.yaml").exists()


def test_evaluate_stage_refuses_without_explicit_protected_approval(tmp_path):
    # Fails closed regardless of the freeze state: no --allow-protected-after-freeze.
    with pytest.raises(SystemExit, match="refused before the model freeze"):
        main(["--stage", "evaluate", "--assemble-only", "--work-root", str(tmp_path),
              "--output", str(tmp_path / "out.yaml")])
    assert not (tmp_path / "out.yaml").exists()


def test_assemble_only_fixture_counts_families_where_p3_exceeds_p1(tmp_path):
    work = tmp_path / "work"
    detect = {family: 1.0 for family in FAMILIES}
    detect["wheel_slip"] = 0.0
    fold_tables(work, p3_detect=detect)
    output = tmp_path / "unseen_family.yaml"
    assert main(["--stage", "evaluate", "--assemble-only", "--engineering-fixture",
                 "--work-root", str(work), "--output", str(output), "--replicates", "20"]) == 0
    report = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert report["complete"] is False and report["research_evidence"] is False
    assert report["families"] == list(FAMILIES)
    assert report["h5"]["count"] == 6
    assert "wheel_slip" not in report["h5"]["families_where_p3_exceeds_p1"]
    assert report["h5"]["pooled_claim"].startswith("none")
    wheel = report["per_family"]["wheel_slip"]
    assert wheel["p3"]["event_recall"] == 0.0
    assert wheel["fold_threshold"] == {"p3": 0.4, "p1": 0.6}
    assert wheel["p3"]["event_count"] == wheel["family_event_count_full_denominator"]
    assert "false_alerts_per_clean_mission_at_fold_threshold" in wheel
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["--stage", "evaluate", "--assemble-only", "--engineering-fixture",
              "--work-root", str(work), "--output", str(output)])


def test_evaluate_folds_marks_complete_only_when_all_folds_exist(tmp_path):
    work = tmp_path / "work"
    fold_tables(work, p3_detect={family: 0.9 for family in FAMILIES})
    report = evaluate_folds(list(FAMILIES), work, bootstrap=BOOTSTRAP, allow_protected=True,
                            gate_passed=True, engineering_fixture=False)
    assert report["complete"] is True
    assert report["h5"]["supported"] in {True, False}
    assert len(report["families_evaluated"]) == 7
    partial = evaluate_folds([*FAMILIES, "unknown_family"], work, bootstrap=BOOTSTRAP,
                             allow_protected=True, gate_passed=True, engineering_fixture=False)
    assert partial["complete"] is False
    assert partial["per_family"]["unknown_family"]["status"] == "missing_fold_tables"
    with pytest.raises(ValueError, match="before confirmatory freeze"):
        evaluate_folds(list(FAMILIES), work, bootstrap=BOOTSTRAP, allow_protected=True,
                       gate_passed=False, engineering_fixture=False)


def test_split_manifest_declares_seven_folds():
    splits = yaml.safe_load((ROOT / "data/manifests/splits.template.yaml").read_text(encoding="utf-8"))
    assert list(splits["unseen_family_folds"]) == list(FAMILIES)

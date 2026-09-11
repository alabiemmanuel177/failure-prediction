import csv
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from confirmatory_fixtures import write_unfrozen_alarm_policy
from src.evaluation.calibrators import Calibrator
from synthetic_prediction_tables import build_rows, write_table

ROOT = Path(__file__).resolve().parents[1]


def confirmatory_gate_passes() -> bool:
    return run("check_readiness.py", "--stage", "confirmatory").returncode == 0


def run(script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
        capture_output=True, text=True,
    )


def read_csv(path):
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


def test_select_apply_calibration_and_alarm_policy_round_trip(tmp_path):
    table = write_table(tmp_path / "p3.csv", build_rows("p3_causal_tcn"))
    report = tmp_path / "calibration.yaml"
    calibrator = tmp_path / "calibrator.json"
    result = run("select_calibration.py", table, "--output", report, "--calibrator-output", calibrator,
                 "--model-id", "p3_causal_tcn")
    assert result.returncode == 0, result.stderr
    document = yaml.safe_load(report.read_text())
    assert document["model_id"] == "p3_causal_tcn"
    assert document["protected_test_used"] is False and document["confirmatory"] is False
    assert document["selection_split"] == "validation"
    assert document["chosen_method"] in document["candidate_methods"]
    assert document["before"]["event_recall_at_budget"] == 0.75
    assert document["constraints"]["event_recall_at_budget_max_drop"] == 0.02
    assert Calibrator.from_json_bytes(calibrator.read_bytes()).sha256() == document["chosen_calibrator_sha256"]
    assert run("select_calibration.py", table, "--output", report, "--calibrator-output", calibrator).returncode != 0

    calibrated = tmp_path / "p3.calibrated.csv"
    result = run("apply_calibration.py", table, calibrator, calibrated)
    assert result.returncode == 0, result.stderr
    rows = read_csv(calibrated)
    assert len(rows) == 180 and set(rows[0]) >= {"raw_score", "risk_score"}
    assert all(0.0 <= float(row["risk_score"]) <= 1.0 for row in rows)
    assert run("apply_calibration.py", table, calibrator, calibrated).returncode != 0
    if document["chosen_method"] != "identity":
        again = run("apply_calibration.py", calibrated, calibrator, tmp_path / "twice.csv")
        assert again.returncode != 0 and "exactly once" in again.stderr

    alarmed = tmp_path / "p3.alarmed.csv"
    result = run("apply_alarm_policy.py", calibrated, alarmed, "--threshold",
                 document["after"]["threshold_at_budget"])
    assert result.returncode == 0, result.stderr
    rows = read_csv(alarmed)
    assert {"alarm", "persistent"} <= set(rows[0])
    alarms = {row["run_id"] for row in rows if row["alarm"] == "true"}
    assert alarms == {"E1", "E2", "E3"}


def test_apply_alarm_policy_refuses_unfrozen_threshold_and_non_validation(tmp_path):
    table = write_table(tmp_path / "p3.csv", build_rows("p3_causal_tcn"))
    alarm = write_unfrozen_alarm_policy(tmp_path / "alarm_policy.yaml")
    assert yaml.safe_load(alarm.read_text())["threshold"] is None
    result = run("apply_alarm_policy.py", table, tmp_path / "out.csv", "--alarm", alarm)
    assert result.returncode != 0 and "not frozen" in result.stderr
    assert not (tmp_path / "out.csv").exists()
    development = write_table(tmp_path / "dev.csv", build_rows("p3_causal_tcn", split="development"))
    result = run("apply_alarm_policy.py", development, tmp_path / "out.csv", "--threshold", "0.5")
    assert result.returncode != 0 and "not validation" in result.stderr
    protected = write_table(tmp_path / "test.csv", build_rows("p3_causal_tcn", split="held_out_map_test"))
    result = run("apply_alarm_policy.py", protected, tmp_path / "out.csv")
    assert result.returncode != 0 and "pass --allow-protected-after-freeze" in result.stderr
    result = run("apply_alarm_policy.py", protected, tmp_path / "out.csv",
                 "--allow-protected-after-freeze", "--threshold", "0.5")
    assert result.returncode != 0
    if confirmatory_gate_passes():
        # Post-freeze: the gate passes, and test-time threshold adaptation stays forbidden.
        assert "forbidden outside validation" in result.stderr
    else:
        assert "confirmatory" in result.stderr
    assert not (tmp_path / "out.csv").exists()


def test_apply_calibration_refuses_protected_rows_without_gate(tmp_path):
    calibrator = tmp_path / "calibrator.json"
    calibrator.write_bytes(Calibrator("identity").to_json_bytes())
    protected = write_table(tmp_path / "test.csv", build_rows("p3_causal_tcn", split="held_out_map_test"))
    result = run("apply_calibration.py", protected, calibrator, tmp_path / "out.csv")
    assert result.returncode != 0 and "explicit approval" in result.stderr
    assert not (tmp_path / "out.csv").exists()
    result = run("apply_calibration.py", protected, calibrator, tmp_path / "out.csv",
                 "--allow-protected-after-freeze")
    if confirmatory_gate_passes():
        # Post-freeze: explicit approval plus the passing gate admits protected rows.
        assert result.returncode == 0, result.stderr
        assert "protected_test_used=true" in result.stdout
        assert len(read_csv(tmp_path / "out.csv")) == 180
    else:
        assert result.returncode != 0 and "confirmatory freeze" in result.stderr
        assert not (tmp_path / "out.csv").exists()


def test_validate_alarm_policy_sweeps_without_selecting(tmp_path):
    table = write_table(tmp_path / "p3.csv", build_rows("p3_causal_tcn"))
    output = tmp_path / "sweep.yaml"
    result = run("validate_alarm_policy.py", table, "--output", output)
    assert result.returncode == 0, result.stderr
    document = yaml.safe_load(output.read_text())
    assert document["automatic_selection"] is False
    assert document["preregistered_setting_retained"] is True
    assert document["preregistered_setting"]["persistence"] == "2-of-3"
    assert document["preregistered_setting"]["cooldown_seconds"] == 10.0
    assert document["preregistered_setting"]["result"]["event_recall"] == 0.75
    assert len(document["sweep"]) == 16
    assert sum(entry["preregistered"] for entry in document["sweep"]) == 1
    one_of_one = next(e for e in document["sweep"] if e["persistence"] == "1-of-1" and e["cooldown_seconds"] == 10.0)
    assert one_of_one["lead_time"]["median_seconds_detected"] == 5.0
    assert run("validate_alarm_policy.py", table, "--output", output).returncode != 0
    development = write_table(tmp_path / "dev.csv", build_rows("p3_causal_tcn", split="development"))
    assert run("validate_alarm_policy.py", development, "--output", tmp_path / "x.yaml").returncode != 0

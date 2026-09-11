import csv
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def write_rows(path, split="validation"):
    rows = []
    for run_id, family, score, event in [
        ("clean", "none", 0.2, None), ("event", "lidar_dropout", 0.9, 5.0)
    ]:
        for decision in (1.0, 2.0, 3.0):
            rows.append({
                "run_id": run_id, "decision_time": decision, "risk_score": score,
                "eligibility": "eligible_negative" if event is None else "eligible_positive",
                "primary_event_time": event, "fault_family": family,
                "split": split, "protected_test_used": "false",
            })
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def test_threshold_cli_writes_validation_only_freeze(tmp_path):
    source = tmp_path / "predictions.csv"
    output = tmp_path / "threshold.yaml"
    write_rows(source)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/select_alarm_threshold.py"),
         str(source), str(output)], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = yaml.safe_load(output.read_text())
    assert report["selection_split"] == "validation"
    assert report["protected_test_used"] is False
    assert report["false_alerts_per_clean_mission"] <= 0.10


def test_threshold_cli_rejects_nonvalidation_rows(tmp_path):
    source = tmp_path / "predictions.csv"
    write_rows(source, split="test")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/select_alarm_threshold.py"),
         str(source), str(tmp_path / "threshold.yaml")], capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "validation rows only" in result.stderr

import csv
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_warning_trace_writes_dependency_free_svg(tmp_path):
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "run_id", "decision_time", "risk_score", "alarm", "primary_event_time",
        ])
        writer.writeheader()
        writer.writerows([
            {"run_id": "r1", "decision_time": 10, "risk_score": 0.1,
             "alarm": "false", "primary_event_time": 20},
            {"run_id": "r1", "decision_time": 15, "risk_score": 0.9,
             "alarm": "true", "primary_event_time": 20},
            {"run_id": "r1", "decision_time": 20, "risk_score": 1.0,
             "alarm": "false", "primary_event_time": 20},
        ])
    output = tmp_path / "trace.svg"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/plot_warning_trace.py"),
        str(predictions), "r1", str(output),
    ], check=True)
    text = output.read_text(encoding="utf-8")
    assert text.startswith("<svg")
    assert "Warning trace" in text
    assert "#b3261e" in text

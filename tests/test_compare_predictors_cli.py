import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from synthetic_prediction_tables import build_rows, write_table

ROOT = Path(__file__).resolve().parents[1]


def run_compare(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/compare_predictors.py"), *map(str, args)],
        capture_output=True, text=True,
    )


def test_compare_predictors_reports_hand_calculated_validation_metrics(tmp_path):
    p3 = write_table(tmp_path / "p3.csv", build_rows("p3_causal_tcn"))
    p1 = write_table(tmp_path / "p1.csv", build_rows("p1_threshold_rules"))
    latency = tmp_path / "p3_latency.json"
    latency.write_text(json.dumps({"median_ms": 1.5, "p95_ms": 3.0, "max_ms": 4.0}))
    output = tmp_path / "validation_comparison.yaml"
    result = run_compare(
        "--table", f"p3_causal_tcn={p3}", "--table", f"p1_threshold_rules={p1}",
        "--latency", f"p3_causal_tcn={latency}", "--replicates", 40, "--paired-replicates", 100,
        "--output", output,
    )
    assert result.returncode == 0, result.stderr
    report = yaml.safe_load(output.read_text())
    assert report["scope"] == "validation_only" and report["confirmatory"] is False
    assert report["protected_test_used"] is False
    assert report["episode_overlap"]["identical_episode_sets"] is True

    tcn = report["models"]["p3_causal_tcn"]
    assert tcn["threshold_selection"]["threshold"] == 0.9
    assert tcn["threshold_selection"]["persistence"] == "2-of-3"
    assert tcn["event_recall"] == 0.75
    assert tcn["detected_event_count"] == 3 and tcn["event_count"] == 4
    assert tcn["false_alerts_per_clean_mission"] == 0.0
    assert tcn["false_alerts_per_non_event_mission"] == 0.0
    assert tcn["lead_time"]["median_seconds_detected"] == 4.5
    assert tcn["lead_time"]["undetected_event_count"] == 1
    assert tcn["latency"]["p95_ms"] == 3.0
    assert tcn["discrimination"]["auprc_bootstrap"]["hierarchy"] == ["map", "route", "episode"]
    assert tcn["discrimination"]["auprc_bootstrap"]["confidence_interval"] is not None
    assert tcn["by_family"]["lidar_dropout"]["event_recall"] == 1.0
    assert tcn["by_family"]["planner_oscillation"]["event_recall"] == 0.0
    assert tcn["by_map"]["m0"]["event_count"] == 2
    assert tcn["alert_burden"]["clean"]["total_alerts"] == 0
    assert 0.0 <= tcn["calibration"]["ece"] <= 1.0

    rules = report["models"]["p1_threshold_rules"]
    assert rules["threshold_selection"]["threshold"] == 1.0
    assert rules["event_recall"] == 0.25
    assert rules["false_alerts_per_clean_mission"] == 0.0
    assert rules["false_alerts_per_non_event_mission"] == pytest.approx(0.2)
    assert rules["false_alert_count"] == 1
    assert rules["lead_time"]["median_seconds_detected"] == 2.5
    assert rules["latency"] is None

    paired = report["paired_primary_minus_baseline"]["result"]
    assert paired["point_estimate"] == pytest.approx(0.5)
    assert paired["event_episode_count"] == 4
    assert paired["contrast"] == "p3_causal_tcn_minus_p1_threshold_rules_event_recall"
    assert run_compare("--table", f"p3_causal_tcn={p3}", "--output", output).returncode != 0


def test_compare_predictors_rejects_non_validation_and_mismatched_ids(tmp_path):
    development = write_table(tmp_path / "dev.csv", build_rows("p3_causal_tcn", split="development"))
    result = run_compare("--table", f"p3_causal_tcn={development}", "--output", tmp_path / "a.yaml")
    assert result.returncode != 0 and "validation rows only" in result.stderr
    p3 = write_table(tmp_path / "p3.csv", build_rows("p3_causal_tcn"))
    result = run_compare("--table", f"p4_gru={p3}", "--output", tmp_path / "b.yaml")
    assert result.returncode != 0 and "model_id" in result.stderr

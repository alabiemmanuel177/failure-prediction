import json
from pathlib import Path

import pytest
import yaml

from confirmatory_fixtures import table_rows, write_table
from scripts.audit_natural_failures import main, natural_failures


def manifest_rows(dataset_id: str, split: str, *, natural: int) -> list[dict]:
    rows = []
    for index in range(6):
        clean_event = index < natural
        rows.append({
            "run_id": f"{dataset_id}-clean-{index}", "split": split, "map_id": f"{split[:3]}_00",
            "route_id": f"{split[:3]}_00_r0", "seed": index, "system_id": "S0",
            "dataset_episode_key": f"k{index}", "fault_family": "none", "severity": "none",
            "primary_event_class": ("collision" if index % 2 else "mission_timeout") if clean_event else None,
            "primary_event_time": 30.0 if clean_event else None,
            "episode_start": 12.0, "episode_end": 60.0, "positive_sequences": 18 if clean_event else 0,
            "protected_test_used": False,
        })
        rows.append({
            "run_id": f"{dataset_id}-lidar-{index}", "split": split, "map_id": f"{split[:3]}_00",
            "route_id": f"{split[:3]}_00_r0", "seed": 100 + index, "system_id": "S0",
            "fault_family": "lidar_dropout", "severity": "medium",
            "primary_event_class": "collision", "primary_event_time": 30.0,
            "episode_start": 12.0, "episode_end": 31.0, "protected_test_used": False,
        })
    return rows


def write_manifest(root: Path, dataset_id: str, rows: list[dict]) -> Path:
    path = root / dataset_id / "extraction_manifest.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_lists_only_no_injection_episodes_with_events():
    rows = manifest_rows("dev", "development", natural=2)
    failures, summary = natural_failures("dev", rows)
    assert [item["run_id"] for item in failures] == ["dev-clean-0", "dev-clean-1"]
    assert summary["clean_episodes"] == 6 and summary["natural_failures"] == 2
    assert summary["by_class"] == {"collision": 1, "mission_timeout": 1}
    assert summary["timeouts"] == 1
    assert failures[0]["seconds_from_episode_start_to_event"] == pytest.approx(18.0)
    rows[0]["protected_test_used"] = True
    with pytest.raises(ValueError, match="protected_test_used"):
        natural_failures("dev", rows)


def test_audit_reports_warning_performance_separately(tmp_path):
    derived = tmp_path / "derived"
    dev_rows = manifest_rows("dev", "development", natural=2)
    val_rows = manifest_rows("val", "validation", natural=1)
    write_manifest(derived, "dev", dev_rows)
    write_manifest(derived, "val", val_rows)
    episodes = [
        {"run_id": row["run_id"], "split": row["split"], "map_id": row["map_id"],
         "route_id": row["route_id"], "fault_family": row["fault_family"], "severity": row["severity"],
         "seed": row["seed"], "primary_event_class": row["primary_event_class"],
         "primary_event_time": row["primary_event_time"]}
        for row in [*dev_rows, *val_rows]
    ]
    table = write_table(tmp_path / "p3.csv", table_rows(
        episodes, model_id="p3_causal_tcn", protected=False, detect_probability=1.0,
        false_alert_probability=0.0, lead_seconds=5.0,
    ))
    output = tmp_path / "natural_failure_audit.yaml"
    assert main(["--derived-root", str(derived), "--dataset-id", "dev", "--dataset-id", "val",
                 "--predictions", str(table), "--output", str(output)]) == 0
    report = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert report["merged_with_injected_results"] is False
    assert report["protected_test_used"] is False
    assert report["totals"]["natural_failures"] == 3
    assert report["totals"]["by_split"] == {"development": 2, "validation": 1}
    assert {item["run_id"] for item in report["natural_failures"]} == {
        "dev-clean-0", "dev-clean-1", "val-clean-0",
    }
    performance = report["warning_performance"][0]
    assert performance["status"] == "evaluated"
    assert performance["natural_failures_in_table"] == 3
    assert performance["natural_failures_excluding_timeouts"]["event_count"] == 1
    assert performance["natural_failures_excluding_timeouts"]["event_recall"] == 1.0
    assert performance["timeouts_analysed_separately"]["count"] == 2
    assert performance["by_class"]["collision"]["lead_time"]["median_seconds_detected"] == pytest.approx(5.0)
    # Injected episodes never enter the audit.
    assert all("lidar" not in item["run_id"] for item in report["natural_failures"])
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["--derived-root", str(derived), "--dataset-id", "dev", "--output", str(output)])


def test_audit_rejects_protected_prediction_tables(tmp_path):
    derived = tmp_path / "derived"
    dev_rows = manifest_rows("dev", "development", natural=1)
    write_manifest(derived, "dev", dev_rows)
    episodes = [{"run_id": "dev-clean-0", "split": "development", "map_id": "dev_00",
                 "route_id": "dev_00_r0", "fault_family": "none", "severity": "none", "seed": 0,
                 "primary_event_class": "mission_timeout", "primary_event_time": 30.0}]
    table = write_table(tmp_path / "p3.csv", table_rows(
        episodes, model_id="p3_causal_tcn", protected=True, detect_probability=1.0,
        false_alert_probability=0.0,
    ))
    with pytest.raises(SystemExit, match="pre-protected tables only"):
        main(["--derived-root", str(derived), "--dataset-id", "dev", "--predictions", str(table),
              "--output", str(tmp_path / "out.yaml")])

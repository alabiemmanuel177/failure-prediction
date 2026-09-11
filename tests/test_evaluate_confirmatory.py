from pathlib import Path

import pytest
import yaml

from confirmatory_fixtures import ROOT, held_out_episodes, table_rows, write_table
from scripts.evaluate_confirmatory import (
    build_report, episode_outcomes, hierarchical_bootstrap, load_table, main, recall_statistic,
)

BOOTSTRAP = {"replicates": 60, "seed": 3}


def alarm_policy() -> dict:
    alarm = yaml.safe_load((ROOT / "configs/alarm_policy.yaml").read_text(encoding="utf-8"))
    alarm["threshold"] = 0.5
    return alarm


def synthetic_tables(tmp_path: Path, *, event_classes=("collision",)) -> dict[str, Path]:
    episodes = held_out_episodes(event_classes=event_classes)
    p3 = table_rows(episodes, model_id="p3_causal_tcn", protected=True, detect_probability=0.9,
                    false_alert_probability=0.05, seed=11, calibrate=lambda value: value ** 2)
    p1 = table_rows(episodes, model_id="p1_threshold_rules", protected=True, detect_probability=0.5,
                    false_alert_probability=0.1, seed=12)
    p4 = table_rows(episodes, model_id="p4_gru", protected=True, detect_probability=0.7,
                    false_alert_probability=0.05, seed=13)
    validation = table_rows(
        held_out_episodes(maps=("val_00",), split="validation"), model_id="p3_causal_tcn",
        protected=False, detect_probability=0.9, false_alert_probability=0.05, seed=14,
        calibrate=lambda value: value ** 2,
    )
    return {
        "p3": write_table(tmp_path / "p3.csv", p3), "p1": write_table(tmp_path / "p1.csv", p1),
        "p4": write_table(tmp_path / "p4.csv", p4),
        "validation": write_table(tmp_path / "validation_p3.csv", validation),
    }


def test_report_computes_paired_h1_with_full_denominators(tmp_path):
    paths = synthetic_tables(tmp_path)
    tables = {key: load_table(paths[key]) for key in ("p3", "p1", "p4")}
    report = build_report(
        tables=tables, threshold=0.5, alarm_policy=alarm_policy(),
        validation_rows=load_table(paths["validation"], require_alarm=False), bootstrap=BOOTSTRAP,
    )
    assert report["complete"] is True and report["protected_test_used"] is True
    assert report["frozen_threshold"] == 0.5
    p3, p1 = report["models"]["p3"], report["models"]["p1"]
    assert p3["event_count"] == p1["event_count"] > 0
    assert p3["event_recall"] > p1["event_recall"]
    assert p3["lead_time"]["event_count_full_denominator"] == p3["event_count"]
    assert p3["lead_time"]["detected_event_count"] + p3["lead_time"]["undetected_event_count"] == p3["event_count"]
    assert p3["lead_time"]["median_seconds_detected"] == pytest.approx(4.0)
    assert set(p3["by_family"]) >= {"none", "camera_occlusion"}
    assert set(p3["by_map"]) == {"test_00", "test_01", "test_02"}
    assert p3["clean_mission_count"] == 12
    h1 = report["h1"]
    assert h1["label"] == "confirmatory"
    assert h1["point_estimate"] == pytest.approx(p3["event_recall"] - p1["event_recall"])
    assert h1["hierarchy"] == ["map", "route", "episode"]
    lower, upper = h1["confidence_interval"]
    assert lower <= h1["point_estimate"] <= upper
    assert h1["denominator_events"] == p3["event_count"]
    assert sum(h1["discordant_pairs"].values()) == p3["event_count"]
    h2 = report["h2"]
    assert h2["label"] == "supporting"
    assert h2["median_lead_seconds"] == pytest.approx(4.0)
    assert h2["point_estimate_meets_minimum"] is True
    assert h2["denominator_events"] == p3["event_count"]
    h3 = report["h3"]
    assert h3["validation"]["brier_before"] != h3["validation"]["brier_after"]
    assert isinstance(h3["supported"], bool)
    assert report["h4"]["status"] == "ablation_reports_absent"
    assert report["exploratory_models"]["p4"]["label"] == "exploratory"
    assert report["timeouts"]["analysed_separately"] is True
    assert report["timeouts"]["p3"]["timeout_episode_count"] == 0


def test_timeouts_are_separated_from_primary_events(tmp_path):
    paths = synthetic_tables(tmp_path, event_classes=("collision", "mission_timeout"))
    tables = {key: load_table(paths[key]) for key in ("p3", "p1")}
    report = build_report(tables=tables, threshold=0.5, alarm_policy=alarm_policy(), bootstrap=BOOTSTRAP)
    timeouts = report["timeouts"]["p3"]
    assert timeouts["timeout_episode_count"] > 0
    primary = report["models"]["p3"]["event_count"]
    assert primary + timeouts["timeout_episode_count"] == timeouts["all_events_including_timeouts"]["event_count"]
    assert report["h1"]["denominator_events"] == primary


def test_hierarchical_bootstrap_is_deterministic_and_episode_unique():
    records = [
        {"run_id": f"e{i}", "map_id": f"m{i % 2}", "route_id": f"r{i % 3}", "has_event": True,
         "detected": i % 3 != 0} for i in range(12)
    ]
    first = hierarchical_bootstrap(records, recall_statistic, replicates=50, seed=1)
    second = hierarchical_bootstrap(records, recall_statistic, replicates=50, seed=1)
    assert first == second
    assert first["point_estimate"] == pytest.approx(8 / 12)
    with pytest.raises(ValueError, match="unique"):
        hierarchical_bootstrap(records + [records[0]], recall_statistic, replicates=5)


def test_cli_refuses_protected_tables_without_approval_and_fixture_is_not_evidence(tmp_path):
    paths = synthetic_tables(tmp_path)
    alarm_path = tmp_path / "alarm.yaml"
    alarm_path.write_text(yaml.safe_dump(alarm_policy()), encoding="utf-8")
    output = tmp_path / "held_out_map.yaml"
    # Protected tables are refused without --allow-protected-after-freeze regardless of
    # whether the workspace freeze exists.
    with pytest.raises(SystemExit, match="refused: refusing protected data access without explicit approval"):
        main(["--p3", str(paths["p3"]), "--p1", str(paths["p1"]), "--alarm", str(alarm_path),
              "--output", str(output), "--replicates", "20"])
    assert not output.exists()
    assert main([
        "--p3", str(paths["p3"]), "--p1", str(paths["p1"]), "--p4", str(paths["p4"]),
        "--validation-p3", str(paths["validation"]), "--alarm", str(alarm_path),
        "--output", str(output), "--engineering-fixture", "--replicates", "20",
    ]) == 0
    report = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert report["complete"] is False
    assert report["research_evidence"] is False
    assert report["engineering_fixture"] is True
    assert report["evidence_status"] == "non_research_engineering_fixture"
    assert report["model_ids"]["p3"] == "p3_causal_tcn"
    assert str(paths["p3"]) in report["inputs_sha256"]
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["--p3", str(paths["p3"]), "--p1", str(paths["p1"]), "--alarm", str(alarm_path),
              "--output", str(output), "--engineering-fixture"])


def test_tables_must_be_alarm_applied_and_paired(tmp_path):
    paths = synthetic_tables(tmp_path)
    rows = load_table(paths["p3"])
    bare = tmp_path / "bare.csv"
    text = paths["p3"].read_text(encoding="utf-8").splitlines()
    header = text[0].split(",")
    keep = [index for index, name in enumerate(header) if name not in {"alarm", "persistent"}]
    bare.write_text("\n".join(",".join(line.split(",")[i] for i in keep) for line in text) + "\n")
    with pytest.raises(ValueError, match="apply_alarm_policy"):
        load_table(bare)
    truncated = [row for row in rows if row["map_id"] != "test_02"]
    with pytest.raises(ValueError, match="same held-out episodes"):
        build_report(tables={"p3": truncated, "p1": load_table(paths["p1"])}, threshold=0.5,
                     alarm_policy=alarm_policy(), bootstrap=BOOTSTRAP)
    outcomes = episode_outcomes(rows)
    assert all(row["lead_seconds"] is None or row["lead_seconds"] >= 0 for row in outcomes.values())

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def read_yaml(relative_path: str):
    return yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8"))


def test_window_geometry_is_causal_and_nonempty():
    config = read_yaml("configs/failure_events.yaml")["windowing"]
    assert config["history_seconds"] > 0
    assert config["warning_horizon_seconds"] > config["too_late_guard_seconds"] > 0
    assert config["negative_guard_seconds"] >= config["warning_horizon_seconds"]
    assert config["decision_stride_seconds"] > 0


def test_alarm_policy_matches_primary_protocol_budget():
    policy = read_yaml("configs/alarm_policy.yaml")
    assert policy["selection_split"] == "validation"
    assert policy["false_alert_budget_per_clean_mission"] == 0.10
    assert policy["test_threshold_adaptation"] == "forbidden"
    assert policy["persistence"]["required_above_threshold"] <= policy["persistence"]["decisions_considered"]


def test_leakage_policy_fails_closed():
    denylist = read_yaml("configs/leakage_denylist.yaml")
    assert denylist["policy"] == "fail_closed"
    forbidden = set(denylist["forbidden_field_patterns"])
    assert {"severity", "actual_onset", "event_time", "terminal_result"} <= forbidden
    assert "future_interpolation" in denylist["forbidden_transformations"]
    assert "/research2/events" in denylist["forbidden_topic_patterns"]
    assert {"run_id", "recorder_loss"} <= forbidden


def test_splits_are_episode_level_and_protected():
    splits = read_yaml("data/manifests/splits.template.yaml")
    assert splits["independent_unit"] == "episode"
    assert splits["split_before_window_extraction"] is True
    assert splits["protected_splits_read_only"] is True
    assert splits["held_out_map_test"]["inspect_only_after_policy_freeze"] is True
    assert splits["development"]["maps"] == [
        "dev_00", "dev_01", "dev_02", "dev_03", "dev_04", "dev_05"
    ]
    assert splits["validation"]["maps"] == ["val_00", "val_01", "val_02"]
    assert splits["held_out_map_test"]["maps"] == []

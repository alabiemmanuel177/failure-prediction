import hashlib
from pathlib import Path
import re
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def read_yaml(relative_path: str):
    return yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8"))


def sha256_of(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def hash_fields(value, path=""):
    """Every ``*_sha256`` leaf (lists included) with its dotted path."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from hash_fields(child, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from hash_fields(child, f"{path}[{index}]")
    elif path.split(".")[-1].split("[")[0].endswith("_sha256"):
        yield path, value


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
    # Assigned strictly after the model freeze and bound to the freeze record's hash.
    held_out = splits["held_out_map_test"]
    assert held_out["maps"] == ["test_00", "test_01", "test_02"]
    assert held_out["status"] == "assigned_after_model_freeze"
    assert len(held_out["routes"]) == 18 and len(set(held_out["routes"])) == 18
    assert all(any(route.startswith(f"{map_id}_r") for map_id in held_out["maps"]) for route in held_out["routes"])
    assert held_out["model_freeze_sha256"] == sha256_of("configs/model_freeze.yaml")
    assert set(held_out["maps"]).isdisjoint(splits["development"]["maps"] + splits["validation"]["maps"])


def test_confirmatory_gate_requires_hash_addressed_model_freeze():
    template = read_yaml("configs/model_freeze.template.yaml")
    assert template["frozen"] is False
    assert template["protected_outcomes_consulted"] is False
    assert template["declaration"]["protected_maps_or_outcomes_inspected"] is False
    assert template["predictor"]["checkpoint_sha256"].startswith("TODO")

    # The signed freeze fills the template hash-for-hash and is internally consistent.
    freeze = read_yaml("configs/model_freeze.yaml")
    alarm = read_yaml("configs/alarm_policy.yaml")
    assert freeze["frozen"] is True
    assert freeze["protected_outcomes_consulted"] is False
    assert freeze["declaration"] == {
        "model_selection_complete": True, "calibration_selection_complete": True,
        "threshold_selection_complete": True, "protected_maps_or_outcomes_inspected": False,
    }
    assert "TODO" not in (ROOT / "configs/model_freeze.yaml").read_text(encoding="utf-8")
    hashes = dict(hash_fields(freeze))
    assert {"predictor.checkpoint_sha256", "calibration.artifact_sha256", "alarm_policy.config_sha256",
            "dataset.split_manifest_sha256", "analysis.analysis_plan_sha256"} <= set(hashes)
    assert all(HEX64.match(str(value)) for value in hashes.values()), hashes
    assert freeze["alarm_policy"]["threshold"] == alarm["threshold"] is not None
    assert freeze["alarm_policy"]["config_sha256"] == sha256_of("configs/alarm_policy.yaml")
    assert freeze["alarm_policy"]["validation_false_alarm_budget"] == alarm["false_alert_budget_per_clean_mission"]
    assert freeze["predictor"]["model_id"] == "p3_causal_tcn"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_readiness.py"),
         "--stage", "confirmatory"],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "READY for confirmatory" in result.stdout

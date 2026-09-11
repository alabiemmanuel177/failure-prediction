"""Concurrency shift check: manifest builder, validator and evaluator (no ROS, no bags)."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest
import yaml

from scripts.build_concurrency_shift_check import (
    CAMPAIGN_ID, SEED_BASE, VARIANTS, build_manifest, collision_findings, existing_seed_sets,
)
from scripts.evaluate_concurrency_shift import (
    check_campaign_id_of, condition_of, deployable_columns, evaluate,
    standardised_mean_difference,
)
from src.evaluation.prediction_tables import PREDICTION_COLUMNS
from src.experiments import (
    expand_balanced_pilot, targeted_execution_order, validate_concurrency_check,
)

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests" / f"{CAMPAIGN_ID}.yaml"
V2_CAMPAIGN_ID = VARIANTS["s3_serialised"]["campaign_id"]
MANIFEST_V2 = ROOT / "data/manifests" / f"{V2_CAMPAIGN_ID}.yaml"
TARGETED = yaml.safe_load((ROOT / "data/manifests/targeted_development_v1.yaml").read_text(encoding="utf-8"))
SPLITS = yaml.safe_load((ROOT / "data/manifests/splits.template.yaml").read_text(encoding="utf-8"))
SCHEMA = yaml.safe_load((ROOT / "configs/feature_schema.yaml").read_text(encoding="utf-8"))
POLICY = yaml.safe_load((ROOT / "configs/concurrency_shift_policy.yaml").read_text(encoding="utf-8"))
ALARM = yaml.safe_load((ROOT / "configs/alarm_policy.yaml").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ manifest
def test_shift_check_design_is_36_episodes_with_fresh_disjoint_seeds():
    document = build_manifest(TARGETED, now_utc="2026-09-03T00:00:00Z", policy_sha256="0" * 64)
    assert validate_concurrency_check(document, SPLITS) == []
    assert collision_findings(document, ROOT / "data/manifests") == []
    episodes = expand_balanced_pilot(document)
    assert len(episodes) == document["expected_episode_count"] == 36
    assert len({item["episode_key"] for item in episodes}) == 36
    seeds = {item["seed"] for item in episodes}
    assert len(seeds) == 36 and min(seeds) == SEED_BASE == 4_000_000
    for name, theirs in existing_seed_sets(ROOT / "data/manifests").items():
        assert seeds.isdisjoint(theirs), name
    assert Counter((item["family"], item["system"]) for item in episodes) == {
        ("none", "s0"): 12, ("none", "s3"): 12, ("planner_oscillation", "s0"): 12,
    }
    assert all(item["map"].startswith("dev_") for item in episodes)
    assert document["execution_policy"]["concurrency"] == 6
    assert document["execution_policy"]["maximum_episodes_per_invocation"] == 12
    assert document["execution_policy"]["minimum_free_space_gib_before_episode"] == 100
    assert document["execution_policy"]["stop_on_first_invalid"] is True
    assert document["purpose"] == "measure the health-feature shift between sequential and six-worker execution"
    assert document["protected_test_used"] is False
    assert document["allowed_splits"] == ["development"]
    assert document["campaign_kind"] == "concurrency_check"
    ordered = targeted_execution_order(episodes)
    for start in range(0, 36, 12):
        wave = ordered[start:start + 12]
        assert len({(item["map"], item["route"]) for item in wave}) == 12
        assert len({item["episode_key"].rsplit("-", 2)[-2] for item in wave}) == 1


def test_written_manifest_is_the_builder_output():
    assert MANIFEST.is_file()
    document = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert document["campaign_id"] == CAMPAIGN_ID
    assert validate_concurrency_check(document, SPLITS) == []
    assert collision_findings(document, ROOT / "data/manifests") == []
    rebuilt = build_manifest(
        TARGETED, now_utc=document["preregistered_utc"],
        policy_sha256=document["pass_rule"]["policy_sha256"],
    )
    assert rebuilt == document
    assert document["pass_rule"]["policy"] == "configs/concurrency_shift_policy.yaml"


def test_s3_serialised_variant_is_the_same_design_under_a_one_s3_cap():
    document = build_manifest(TARGETED, now_utc="2026-09-04T00:00:00Z", policy_sha256="0" * 64,
                              variant="s3_serialised")
    assert validate_concurrency_check(document, SPLITS) == []
    assert collision_findings(document, ROOT / "data/manifests") == []
    assert document["campaign_id"] == V2_CAMPAIGN_ID == "concurrency_shift_check_v2"
    assert document["purpose"] == "six workers with at most one S3 episode at a time"
    assert document["pass_rule"]["report"] == "reports/integrity/concurrency_shift_check_v2.yaml"
    assert document["pass_rule"]["policy"] == "configs/concurrency_shift_policy.yaml"
    assert document["execution_policy"]["concurrency"] == 6
    assert document["execution_policy"]["system_concurrency"] == {"s3": 1}
    assert document["execution_policy"]["maximum_episodes_per_invocation"] == 12
    assert document["preceding_check"]["passed"] is False
    v1 = build_manifest(TARGETED, now_utc="x", policy_sha256="y")
    assert v1["design"]["map_routes"] == document["design"]["map_routes"]
    assert v1["design"]["conditions"] == document["design"]["conditions"]
    assert v1["episode_defaults"] == document["episode_defaults"]
    assert "system_concurrency" not in v1["execution_policy"]
    episodes = expand_balanced_pilot(document)
    assert len(episodes) == document["expected_episode_count"] == 36
    assert Counter((item["family"], item["system"]) for item in episodes) == {
        ("none", "s0"): 12, ("none", "s3"): 12, ("planner_oscillation", "s0"): 12,
    }
    seeds = {item["seed"] for item in episodes}
    assert len(seeds) == 36 and min(seeds) == document["design"]["seed_base"] == 4_500_000
    v1_seeds = {item["seed"] for item in expand_balanced_pilot(v1)}
    assert seeds.isdisjoint(v1_seeds)
    for name, theirs in existing_seed_sets(ROOT / "data/manifests", exclude_campaign_id=V2_CAMPAIGN_ID).items():
        assert seeds.isdisjoint(theirs), name
    assert "concurrency_shift_check_v1" in existing_seed_sets(
        ROOT / "data/manifests", exclude_campaign_id=V2_CAMPAIGN_ID,
    )
    # The requested base 5,000,000 is not fresh under wide_v1: map index 1 of v1 starts
    # there, and map index 5 would reach the reserved held-out base 10,000,000.
    requested = json.loads(json.dumps(document))
    requested["design"]["seed_base"] = 5_000_000
    findings = collision_findings(requested, ROOT / "data/manifests")
    assert any("collide with concurrency_shift_check_v1" in item for item in findings)
    assert any("reserved held_out_map_v1 base" in item for item in findings)
    with pytest.raises(ValueError, match="unknown variant"):
        build_manifest(TARGETED, now_utc="x", policy_sha256="y", variant="gpu_only")


def test_written_v2_manifest_is_the_builder_output():
    assert MANIFEST_V2.is_file()
    document = yaml.safe_load(MANIFEST_V2.read_text(encoding="utf-8"))
    assert validate_concurrency_check(document, SPLITS) == []
    assert collision_findings(document, ROOT / "data/manifests") == []
    rebuilt = build_manifest(
        TARGETED, now_utc=document["preregistered_utc"],
        policy_sha256=document["pass_rule"]["policy_sha256"], variant="s3_serialised",
    )
    assert rebuilt == document
    v1 = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert document["pass_rule"]["policy_sha256"] == v1["pass_rule"]["policy_sha256"]
    ordered = targeted_execution_order(expand_balanced_pilot(document))
    for start in range(0, 36, 12):
        wave = ordered[start:start + 12]
        assert len({(item["map"], item["route"]) for item in wave}) == 12
        assert len({item["episode_key"].rsplit("-", 2)[-2] for item in wave}) == 1


def test_validator_checks_the_system_concurrency_block():
    document = build_manifest(TARGETED, now_utc="x", policy_sha256="y", variant="s3_serialised")

    def with_caps(caps):
        return {**document, "execution_policy": {**document["execution_policy"], "system_concurrency": caps}}

    assert validate_concurrency_check(with_caps({"s3": 1}), SPLITS) == []
    assert validate_concurrency_check(with_caps({"s3": 6, "s0": 2}), SPLITS) == []
    assert any("no condition uses" in item for item in validate_concurrency_check(with_caps({"s2": 1}), SPLITS))
    assert any("exceeds the declared concurrency" in item
               for item in validate_concurrency_check(with_caps({"s3": 7}), SPLITS))
    assert any("positive integer" in item for item in validate_concurrency_check(with_caps({"s3": 0}), SPLITS))
    assert any("positive integer" in item for item in validate_concurrency_check(with_caps({"s3": "1"}), SPLITS))
    assert any("must be a mapping" in item for item in validate_concurrency_check(with_caps(["s3"]), SPLITS))


def test_validator_rejects_sequential_or_wrong_designs():
    document = build_manifest(TARGETED, now_utc="x", policy_sha256="y")
    sequential = {**document, "execution_policy": {**document["execution_policy"], "concurrency": 1}}
    assert "concurrency check must declare execution_policy.concurrency of at least 2" in (
        validate_concurrency_check(sequential, SPLITS)
    )
    wrong_family = json.loads(json.dumps(document))
    wrong_family["design"]["conditions"][2]["family"] = "wheel_slip"
    assert any("exactly clean s0, clean s3 and medium planner_oscillation" in item
               for item in validate_concurrency_check(wrong_family, SPLITS))
    validation_split = {**document, "allowed_splits": ["validation"]}
    assert "concurrency check must remain development-only" in validate_concurrency_check(validation_split, SPLITS)
    wrong_kind = {**document, "campaign_kind": "development_supplement"}
    assert "campaign_kind must be concurrency_check" in validate_concurrency_check(wrong_kind, SPLITS)


# ----------------------------------------------------------------- evaluator
COLUMNS = deployable_columns(SCHEMA)
DECISIONS = 24


def write_dataset(root: Path, episodes: list[dict], *, seed: int, latency_offset: dict | None = None,
                  telemetry_rate_hz: float = 10.0) -> None:
    """Synthetic derived dataset in the extract_dataset_sequences layout."""
    rng = random.Random(seed)
    latency_offset = latency_offset or {}
    for step in ("labels", "window_features", "telemetry"):
        (root / step).mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for episode in episodes:
        run_id = episode["run_id"]
        perception = episode["system_id"] == "S3"
        with (root / "labels" / f"{run_id}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["run_id", "decision_index", "decision_time", "window_start", "window_end",
                             "label", "eligibility", "primary_event_class", "primary_event_time"])
            for index in range(DECISIONS):
                time = 10.0 + 0.5 * index
                eligibility = "excluded_near_event_or_injection" if index >= DECISIONS - 2 else "eligible_negative"
                writer.writerow([run_id, index, time, time - 5.0, time,
                                 0 if eligibility == "eligible_negative" else -1, eligibility, "", ""])
        with (root / "window_features" / f"{run_id}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["run_id", "decision_index", "decision_time", *COLUMNS])
            for index in range(DECISIONS):
                row = [run_id, index, 10.0 + 0.5 * index]
                for column in COLUMNS:
                    if column.endswith("__missing"):
                        value = 1 if (column.startswith(("confidence_mean", "uncertainty_mean", "inference_latency_ms")) and not perception) else 0
                    elif column.endswith("__age_seconds"):
                        value = 0.02 + 0.03 * rng.random()
                    elif column == "inference_latency_ms":
                        value = (20.0 + 2.0 * rng.gauss(0, 1) + latency_offset.get(run_id, 0.0)) if perception else 0.0
                    elif column == "valid_return_fraction":
                        value = 0.98 + 0.01 * rng.random()
                    else:
                        value = rng.gauss(0, 1)
                    row.append(value)
                writer.writerow(row)
        with (root / "telemetry" / f"{run_id}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["run_id", "timestamp", "feature", "source", "value", "max_age_seconds"])
            for source, feature in POLICY["message_rate_proxies"]["sources"].items():
                if source == "/research2/features/perception" and not perception:
                    continue
                stamp = 5.0
                while stamp < 10.0 + 0.5 * DECISIONS + 1.0:
                    writer.writerow([run_id, round(stamp, 3), feature, source, 0.0, 1.0])
                    stamp += 1.0 / telemetry_rate_hz
        manifest_rows.append({
            "run_id": run_id, "split": "development", "protected_test_used": False,
            "map_id": episode["map_id"], "route_id": episode["route_id"],
            "fault_family": episode["fault_family"], "severity": episode["severity"],
            "system_id": episode["system_id"], "seed": episode["seed"],
        })
    (root / "extraction_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows), encoding="utf-8",
    )


def synthetic_episodes(prefix: str, per_condition: int) -> list[dict]:
    episodes = []
    conditions = [("none", "none", "S0"), ("none", "none", "S3"), ("planner_oscillation", "medium", "S0")]
    for family, severity, system in conditions:
        for index in range(per_condition):
            episodes.append({
                "run_id": f"{prefix}-{family}-{system}-{index}", "map_id": f"dev_0{index % 6}",
                "route_id": f"dev_0{index % 6}_r{index % 2}", "fault_family": family,
                "severity": severity, "system_id": system, "seed": index,
            })
    return episodes


def write_predictions(path: Path, episodes: list[dict], spikes: set[str]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(PREDICTION_COLUMNS))
        writer.writeheader()
        for episode in episodes:
            for index in range(DECISIONS):
                time = 10.0 + 0.5 * index
                spike = episode["run_id"] in spikes and 6 <= index <= 8
                writer.writerow({
                    "run_id": episode["run_id"], "decision_index": index, "decision_time": time,
                    "split": "development", "map_id": episode["map_id"], "route_id": episode["route_id"],
                    "fault_family": episode["fault_family"], "severity": episode["severity"],
                    "seed": episode["seed"], "protected_test_used": "false",
                    "eligibility": "eligible_negative", "label": 0, "primary_event_class": "",
                    "primary_event_time": "", "model_id": "p3_causal_tcn",
                    "raw_score": 0.9 if spike else 0.1, "risk_score": 0.9 if spike else 0.1,
                })
    return path


def relaxed_policy() -> dict:
    policy = json.loads(json.dumps(POLICY))
    policy["minimum_episodes_per_condition"] = 2
    return policy


def test_condition_matching_and_columns():
    assert condition_of({"fault_family": "none", "system_id": "S0"}) == "clean_s0"
    assert condition_of({"fault_family": "none", "system_id": "s3"}) == "clean_s3"
    assert condition_of({"fault_family": "planner_oscillation", "system_id": "S0"}) == "oscillation"
    assert condition_of({"fault_family": "lidar_dropout", "system_id": "S0"}) is None
    assert len(COLUMNS) == 84
    assert "inference_latency_ms__age_seconds" in COLUMNS and "valid_return_fraction" in COLUMNS


def test_smd_edge_cases():
    assert standardised_mean_difference(np.array([1.0, 1.0]), np.array([1.0, 1.0, 1.0])) == 0.0
    assert standardised_mean_difference(np.array([1.0, 1.0]), np.array([2.0, 2.0])) == float("inf")
    value = standardised_mean_difference(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
    assert value == pytest.approx(1.0)


def test_identical_distributions_pass_and_shifted_latency_fails(tmp_path):
    reference = synthetic_episodes("ref", 4)
    check = synthetic_episodes("chk", 2)
    write_dataset(tmp_path / "ref", reference, seed=1)
    write_dataset(tmp_path / "chk", check, seed=2)
    threshold = tmp_path / "threshold.yaml"
    threshold.write_text(yaml.safe_dump({"threshold": 0.5}), encoding="utf-8")
    predictions = write_predictions(tmp_path / "predictions.csv", reference + check, spikes=set())
    report = evaluate(
        check_root=tmp_path / "chk", reference_root=tmp_path / "ref", policy=relaxed_policy(),
        schema=SCHEMA, check_predictions=predictions, threshold_record=threshold, alarm_config=ALARM,
    )
    assert report["passed"] is True and report["findings"] == []
    assert report["counts"]["check_by_condition"] == {"clean_s0": 2, "clean_s3": 2, "oscillation": 2}
    assert report["counts"]["reference_by_condition"] == {"clean_s0": 4, "clean_s3": 4, "oscillation": 4}
    assert report["counts"]["deployable_columns"] == 84
    latency = report["feature_shift"]["clean_s3"]["features"]["inference_latency_ms"]
    assert abs(latency["smd"]) < 0.5 and latency["highlighted"] is True
    assert set(latency["quantile_shift"]) == {"q05", "q25", "q50", "q75", "q95"}
    assert latency["reference"]["n"] == 4 * (DECISIONS - 2) and latency["check"]["n"] == 2 * (DECISIONS - 2)
    age = report["feature_shift"]["clean_s0"]["features"]["odom_x__age_seconds"] if "odom_x__age_seconds" in COLUMNS else None
    assert age is None  # odom_x is not deployable; only primary features are compared
    assert report["feature_shift"]["clean_s0"]["features"]["command_linear__age_seconds"]["highlighted"] is True
    assert report["feature_shift"]["clean_s0"]["features"]["jerk"]["highlighted"] is False
    assert "quantile_shift" not in report["feature_shift"]["clean_s0"]["features"]["jerk"]
    rates = report["message_rate_shift"]["clean_s3"]["rates_hz"]
    assert rates["/odom"]["reference"]["mean"] == pytest.approx(10.0, abs=0.2)
    assert rates["/research2/features/perception"]["check"]["n"] == 2
    assert report["message_rate_shift"]["clean_s0"]["rates_hz"]["/research2/features/perception"]["check"]["mean"] == 0.0
    alerts = report["false_alerts"]
    assert alerts["within_budget"] is True and alerts["threshold"] == 0.5
    assert alerts["check"]["clean_missions"] == 4 and alerts["check"]["false_alerts_per_clean_mission"] == 0.0
    assert alerts["sequential_reference"]["clean_missions"] == 8
    assert report["protected_test_used"] is False and report["training_performed"] is False

    shifted = synthetic_episodes("shift", 2)
    write_dataset(tmp_path / "shift", shifted, seed=3,
                  latency_offset={item["run_id"]: 40.0 for item in shifted if item["system_id"] == "S3"})
    report = evaluate(
        check_root=tmp_path / "shift", reference_root=tmp_path / "ref", policy=relaxed_policy(),
        schema=SCHEMA,
    )
    assert report["passed"] is False
    assert any("clean_s3: deployable features shifted beyond 0.5 SMD: inference_latency_ms" in item
               for item in report["findings"])
    assert report["feature_shift"]["clean_s3"]["features_beyond_maximum"] == ["inference_latency_ms"]
    assert report["feature_shift"]["clean_s0"]["features_beyond_maximum"] == []
    assert report["false_alerts"] is None
    assert any("no frozen prediction table" in item for item in report["findings"])


def test_false_alert_budget_is_enforced_on_check_clean_missions(tmp_path):
    reference = synthetic_episodes("ref", 3)
    check = synthetic_episodes("chk", 2)
    write_dataset(tmp_path / "ref", reference, seed=5)
    write_dataset(tmp_path / "chk", check, seed=6)
    threshold = tmp_path / "threshold.yaml"
    threshold.write_text(yaml.safe_dump({"threshold": 0.5}), encoding="utf-8")
    spiking = {"chk-none-S0-0"}
    predictions = write_predictions(tmp_path / "predictions.csv", reference + check, spikes=spiking)
    report = evaluate(
        check_root=tmp_path / "chk", reference_root=tmp_path / "ref", policy=relaxed_policy(),
        schema=SCHEMA, check_predictions=predictions, threshold_record=threshold, alarm_config=ALARM,
    )
    alerts = report["false_alerts"]
    assert alerts["check"]["false_alerts"] == 1 and alerts["check"]["clean_missions"] == 4
    assert alerts["check"]["false_alerts_per_clean_mission"] == 0.25
    assert alerts["check"]["per_mission"]["chk-none-S0-0"] == 1
    assert alerts["sequential_reference"]["false_alerts_per_clean_mission"] == 0.0
    assert alerts["difference_per_clean_mission"] == 0.25
    assert alerts["within_budget"] is False and report["passed"] is False
    assert any("exceed the 0.1 budget" in item for item in report["findings"])
    # per-condition breakdown: the alert sits in clean S0; the pass rule stays pooled
    assert alerts["pass_rule_scope"] == "pooled_clean_missions"
    by_condition = alerts["by_condition"]
    assert set(by_condition) == {"clean_s0", "clean_s3"}
    assert by_condition["clean_s0"]["check"] == {
        "clean_missions": 2, "false_alerts": 1, "false_alerts_per_clean_mission": 0.5,
        "missions_without_predictions": [],
    }
    assert by_condition["clean_s3"]["check"]["false_alerts"] == 0
    assert by_condition["clean_s3"]["check"]["clean_missions"] == 2
    assert by_condition["clean_s0"]["sequential_reference"]["clean_missions"] == 3
    assert by_condition["clean_s0"]["difference_per_clean_mission"] == 0.5
    assert by_condition["clean_s0"]["check_within_budget"] is False
    assert by_condition["clean_s3"]["check_within_budget"] is True
    # the synthetic root "chk" names no campaign, so the policy's id is the fallback
    assert report["check_campaign_id"] == "concurrency_shift_check_v1"


def test_evaluator_refuses_protected_or_non_development_inventories(tmp_path):
    episodes = synthetic_episodes("ref", 2)
    write_dataset(tmp_path / "ref", episodes, seed=1)
    path = tmp_path / "ref/extraction_manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["split"] = "held_out_map_test"
    rows[0]["protected_test_used"] = True
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing non-development or protected"):
        evaluate(check_root=tmp_path / "ref", reference_root=tmp_path / "ref",
                 policy=relaxed_policy(), schema=SCHEMA)


def test_check_campaign_id_follows_the_dataset_id():
    assert check_campaign_id_of("concurrency_shift_check_v2-development-36", POLICY) == "concurrency_shift_check_v2"
    assert check_campaign_id_of("concurrency_shift_check_v1-development-36", POLICY) == "concurrency_shift_check_v1"
    assert check_campaign_id_of(None, POLICY) == POLICY["check_campaign_id"] == "concurrency_shift_check_v1"
    assert check_campaign_id_of("odd", POLICY) == "concurrency_shift_check_v1"


def test_cli_evaluates_the_v2_dataset_with_the_unchanged_policy(tmp_path):
    reference = synthetic_episodes("ref", 2)
    check = synthetic_episodes("chk", 2)
    derived = tmp_path / "derived"
    write_dataset(derived / "balanced_pilot_v1-development-648", reference, seed=1)
    write_dataset(derived / "concurrency_shift_check_v2-development-36", check, seed=2)
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(relaxed_policy()), encoding="utf-8")
    output = tmp_path / "reports/concurrency_shift_check_v2.yaml"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/evaluate_concurrency_shift.py"),
        "--check-dataset-id", "concurrency_shift_check_v2-development-36",
        "--derived-root", str(derived), "--policy", str(policy_path), "--output", str(output),
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 1, result.stderr  # no predictions supplied -> not passed
    report = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert report["check_dataset_id"] == "concurrency_shift_check_v2-development-36"
    assert report["check_campaign_id"] == "concurrency_shift_check_v2"
    assert report["policy_check_campaign_id"] == "concurrency_shift_check_v1"
    assert report["pass_rule"]["maximum_absolute_smd"] == 0.5
    assert report["pass_rule"]["false_alert_budget_per_clean_mission"] == 0.1
    assert report["reference_dataset_id"] == "balanced_pilot_v1-development-648"


def test_cli_writes_immutable_report(tmp_path):
    reference = synthetic_episodes("ref", 2)
    check = synthetic_episodes("chk", 2)
    derived = tmp_path / "derived"
    write_dataset(derived / "balanced_pilot_v1-development-648", reference, seed=1)
    write_dataset(derived / "concurrency_shift_check_v1-development-36", check, seed=2)
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(relaxed_policy()), encoding="utf-8")
    output = tmp_path / "reports/concurrency_shift_check_v1.yaml"
    command = [
        sys.executable, str(ROOT / "scripts/evaluate_concurrency_shift.py"),
        "--check-dataset-id", "concurrency_shift_check_v1-development-36",
        "--derived-root", str(derived), "--policy", str(policy_path), "--output", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 1, result.stderr  # no predictions supplied -> not passed
    report = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["inputs"]["policy_sha256"]
    assert report["check_dataset_id"] == "concurrency_shift_check_v1-development-36"
    assert report["reference_dataset_id"] == "balanced_pilot_v1-development-648"
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode != 0 and "refusing to overwrite" in second.stderr

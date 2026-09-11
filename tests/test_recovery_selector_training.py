import csv
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from src.recovery import GuardConfig
from src.recovery.guards import ACTIONS, eligible_actions
from src.recovery.selector_training import (
    FEATURE_NAMES, features_from_row, load_selector_model, predict_action_costs,
    save_selector_model, state_from_row, train_selector,
)


ROOT = Path(__file__).resolve().parents[1]


def synthetic_rows(count=30, split="development", seed=1):
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        state = {
            "stopped": "true", "stop_allowed": "true",
            "localisation_poor": str(rng.random() < 0.5).lower(),
            "planning_stale_or_blocked": str(rng.random() < 0.5).lower(),
            "rear_clearance_m": rng.choice(["0.2", "1.0", ""]), "rotation_clearance_m": "1.0",
            "immediate_collision_risk": "false", "obstruction_may_be_transient": "true",
            "relocalisation_available": "true", "repeated_recovery_count": "0",
        }
        guards = eligible_actions(state_from_row(state), GuardConfig())
        for action in ACTIONS:
            rows.append({
                "split": split, "map_id": "dev_00", "route_id": "dev_00_r0", "seed": str(index),
                "fault_family": "lidar_dropout", "severity": "medium", "warning_id": f"w{index}",
                "risk_score": f"{rng.uniform(0.5, 1.0):.3f}", "diagnosed_signal_group": "planning",
                **state, "candidate_action": action,
                "guard_eligible": str(guards[action][0]).lower(),
                "observed_cost": f"{rng.uniform(0, 60):.2f}",
            })
    return rows


def test_trainer_fits_only_guard_eligible_development_rows_and_predicts_every_action():
    rows = synthetic_rows()
    model = train_selector(rows, GuardConfig())
    assert model["row_counts"]["guard_rejected_excluded"] > 0
    assert model["row_counts"]["fitted"] + model["row_counts"]["guard_rejected_excluded"] == len(rows)
    assert model["feature_names"] == list(FEATURE_NAMES)
    costs = predict_action_costs(model, features_from_row(rows[0]))
    assert set(costs) == set(ACTIONS)
    assert model["protected_test_used"] is False


def test_trainer_asserts_table_agrees_with_frozen_guard():
    rows = synthetic_rows()
    rejected = next(row for row in rows if row["guard_eligible"] == "false")
    rejected["guard_eligible"] = "true"
    with pytest.raises(AssertionError, match="disagrees with the frozen guard"):
        train_selector(rows, GuardConfig())


def test_trainer_refuses_protected_rows_and_requires_development():
    with pytest.raises(ValueError, match="protected"):
        train_selector(synthetic_rows(split="held_out_map_test"), GuardConfig())
    with pytest.raises(ValueError, match="development"):
        train_selector(synthetic_rows(split="validation"), GuardConfig())


def test_model_round_trips_with_sha256_sidecar(tmp_path):
    model = train_selector(synthetic_rows(), GuardConfig())
    path = tmp_path / "selector.json"
    digest = save_selector_model(model, path)
    assert len(digest) == 64
    assert path.with_suffix(".json.sha256").read_text().startswith(digest)
    assert load_selector_model(path)["model_id"] == model["model_id"]
    with pytest.raises(FileExistsError):
        save_selector_model(model, path)
    path.write_text(json.dumps({**model, "actions": {}}))
    with pytest.raises(ValueError, match="checksum"):
        load_selector_model(path)


def test_cli_trains_from_csv_and_reports_validation_regret(tmp_path):
    table = tmp_path / "costs.csv"
    rows = synthetic_rows()
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    evaluation = tmp_path / "validation.csv"
    validation_rows = synthetic_rows(count=10, split="validation", seed=7)
    with evaluation.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(validation_rows[0].keys()))
        writer.writeheader(); writer.writerows(validation_rows)
    output = tmp_path / "model.json"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/train_recovery_selector.py"), str(table),
        "--output", str(output), "--evaluation-table", str(evaluation),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    model = load_selector_model(output)
    assert model["inputs"]["cost_table_sha256"]
    assert model["validation_regret"]["warning_count"] == 10

from pathlib import Path

import pytest
import yaml

from src.recovery.costs import (
    CostWeights, RecoveryOutcome, cost_breakdown, load_cost_config, load_cost_weights,
    observed_cost, outcome_from_row,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_config_has_numeric_weights_in_dominance_order():
    document = load_cost_config(ROOT / "configs/recovery_costs.yaml")
    assert document["status"] == "numeric_weights_proposed_pending_validation_replay"
    assert document["dominance_order"] == [
        "collision", "mission_abort", "failed_recovery", "excessive_delay",
        "path_overhead", "unnecessary_intervention",
    ]
    weights = load_cost_weights(ROOT / "configs/recovery_costs.yaml")
    assert weights.collision == 100.0 and weights.mission_abort == 40.0
    assert weights.failed_recovery == 20.0 and weights.unnecessary_intervention == 5.0
    assert weights.excessive_delay_cap == 15.0 and weights.path_overhead_cap == 10.0


def test_observed_cost_caps_overheads_and_sums_terms():
    weights = load_cost_weights()
    outcome = RecoveryOutcome(collision=True, mission_abort=False, failed_recovery=True,
                              added_time_seconds=100.0, added_path_length_m=50.0,
                              unnecessary_intervention=True)
    breakdown = cost_breakdown(outcome, weights)
    assert breakdown["excessive_delay"] == 15.0 and breakdown["path_overhead"] == 10.0
    assert observed_cost(outcome, weights) == 100 + 20 + 15 + 10 + 5
    modest = RecoveryOutcome(False, False, False, 4.0, 2.5, False)
    assert observed_cost(modest, weights) == pytest.approx(2.0 + 2.5)


def test_weights_must_respect_dominance_order():
    with pytest.raises(ValueError, match="dominance"):
        CostWeights(100, 40, 45, 0.5, 15, 1.0, 10, 5).validate()
    with pytest.raises(ValueError, match="positive"):
        CostWeights(100, 40, 20, 0.5, 15, 1.0, 10, 0).validate()


def test_outcome_from_paired_recovery_row_uses_mission_complete_as_abort_default():
    outcome = outcome_from_row({"collision": "false", "mission_complete": "false",
                                "added_time_seconds": "3", "added_path_length_m": "1"})
    assert outcome.mission_abort is True and outcome.collision is False


def test_config_rejects_missing_selector_training_rule(tmp_path):
    path = tmp_path / "costs.yaml"
    path.write_text(yaml.safe_dump({"numeric_weights": {}, "selector_training_rule": "other"}))
    with pytest.raises(ValueError, match="guard-rejected"):
        load_cost_config(path)

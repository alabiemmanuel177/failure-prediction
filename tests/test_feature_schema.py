from pathlib import Path

import pytest

from src.features import (
    load_primary_feature_set, load_raw_feature_contract, resolve_feature_specs,
)


ROOT = Path(__file__).resolve().parents[1]


def test_raw_contract_covers_cross_profile_perception_and_is_stable_when_absent():
    contract = load_raw_feature_contract(ROOT / "configs/feature_schema.yaml")
    assert contract["confidence_mean"]["sources"] == (
        "/research2/features/perception", "/semantic/confidence"
    )
    specs, grouped = resolve_feature_specs(contract, [])
    assert grouped == {}
    assert [spec.name for spec in specs] == list(contract)
    assert next(spec for spec in specs if spec.name == "confidence_mean").source \
        == "/research2/features/perception"


def test_raw_contract_accepts_one_legacy_source_but_rejects_mixed_sources():
    contract = load_raw_feature_contract(ROOT / "configs/feature_schema.yaml")
    legacy = {
        "feature": "confidence_mean", "source": "/semantic/confidence",
        "max_age_seconds": "1.0",
    }
    specs, _ = resolve_feature_specs(contract, [legacy])
    assert next(spec for spec in specs if spec.name == "confidence_mean").source \
        == "/semantic/confidence"
    compact = {**legacy, "source": "/research2/features/perception"}
    with pytest.raises(ValueError, match="mixed transport sources"):
        resolve_feature_specs(contract, [legacy, compact])


def test_raw_contract_rejects_unknown_features_and_age_drift():
    contract = load_raw_feature_contract(ROOT / "configs/feature_schema.yaml")
    with pytest.raises(ValueError, match="outside raw_feature_contract"):
        resolve_feature_specs(contract, [{
            "feature": "terminal_result", "source": "/diagnostics",
            "max_age_seconds": "1.0",
        }])
    with pytest.raises(ValueError, match="maximum age differs"):
        resolve_feature_specs(contract, [{
            "feature": "command_linear", "source": "/cmd_vel",
            "max_age_seconds": "999",
        }])


def test_primary_feature_set_is_declared_and_excludes_absolute_map_coordinates():
    features = load_primary_feature_set(ROOT / "configs/feature_schema.yaml")
    assert "tracking_error" in features
    assert "replan_rate" in features
    assert "odom_x" not in features
    assert "amcl_x" not in features
    assert "temporal_disagreement" not in features

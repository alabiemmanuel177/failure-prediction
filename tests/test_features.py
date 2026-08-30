from pathlib import Path

import pytest

from src.features import (
    FeatureSpec, LeakageError, LeakagePolicy, NormalizationBundle, ScalarSample,
    extract_decision_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def policy():
    return LeakagePolicy.from_yaml(ROOT / "configs/leakage_denylist.yaml")


def test_resampling_uses_latest_past_sample_and_expires_stale_values():
    rows = extract_decision_rows(
        run_id="r1", decision_times=[1.0, 2.0, 4.0],
        specs=[FeatureSpec("scan_valid_fraction", "/scan", 1.0)],
        samples_by_feature={"scan_valid_fraction": [
            ScalarSample(0.5, 0.9), ScalarSample(2.0, 0.8), ScalarSample(3.5, 0.7),
        ]}, leakage_policy=policy(),
    )
    assert rows[0]["scan_valid_fraction"] == 0.9
    assert rows[0]["scan_valid_fraction__age_seconds"] == 0.5
    assert rows[1]["scan_valid_fraction"] == 0.8
    assert rows[1]["__audit_source_time__scan_valid_fraction"] == 2.0
    assert rows[2]["scan_valid_fraction"] == 0.7


def test_missing_and_stale_are_explicit_not_silently_filled():
    rows = extract_decision_rows(
        run_id="r1", decision_times=[1.0, 3.0],
        specs=[FeatureSpec("pose_covariance_trace", "/amcl_pose", 1.0)],
        samples_by_feature={"pose_covariance_trace": [ScalarSample(0.0, 2.0)]},
        leakage_policy=policy(),
    )
    assert rows[0]["pose_covariance_trace__missing"] == 0
    assert rows[1]["pose_covariance_trace"] == 0.0
    assert rows[1]["pose_covariance_trace__missing"] == 1
    assert rows[1]["__audit_source_time__pose_covariance_trace"] is None


def test_future_only_sample_is_never_selected():
    rows = extract_decision_rows(
        run_id="r1", decision_times=[1.0], specs=[FeatureSpec("speed", "/odom", 1.0)],
        samples_by_feature={"speed": [ScalarSample(1.1, 5.0)]}, leakage_policy=policy(),
    )
    assert rows[0]["speed__missing"] == 1


def test_forbidden_event_and_outcome_fields_fail_closed():
    with pytest.raises(LeakageError):
        extract_decision_rows(
            run_id="r1", decision_times=[1.0],
            specs=[FeatureSpec("risk", "/research2/events", 1.0)],
            samples_by_feature={}, leakage_policy=policy(),
        )
    with pytest.raises(LeakageError):
        extract_decision_rows(
            run_id="r1", decision_times=[1.0],
            specs=[FeatureSpec("terminal_result", "/diagnostics", 1.0)],
            samples_by_feature={}, leakage_policy=policy(),
        )


def test_normalization_fits_development_only_and_preserves_missing_zero():
    rows = [
        {"x": 1.0, "x__missing": 0}, {"x": 3.0, "x__missing": 0},
        {"x": 0.0, "x__missing": 1},
    ]
    bundle = NormalizationBundle.fit(rows, ["x"], split="development")
    assert bundle.transform(rows[0])["x"] == -1.0
    assert bundle.transform(rows[2])["x"] == 0.0
    assert len(bundle.sha256) == 64
    with pytest.raises(ValueError):
        NormalizationBundle.fit(rows, ["x"], split="validation")

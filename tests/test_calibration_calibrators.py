import json

import numpy as np
import pytest
import yaml

from src.evaluation import calibrators
from src.evaluation.calibrators import (
    Calibrator, apply_calibrator_rows, fit_calibrator, fit_isotonic_regression,
    fit_platt_scaling, fit_temperature_scaling, pool_adjacent_violators, select_calibration,
)
from synthetic_prediction_tables import build_rows

ROOT_ALARM = {
    "false_alert_budget_per_clean_mission": 0.10,
    "persistence": {"required_above_threshold": 2, "decisions_considered": 3},
    "cooldown_seconds": 10.0,
}
POLICY = {
    "selection_split": "validation",
    "candidate_methods": ["identity", "temperature_scaling", "platt_scaling", "isotonic_regression"],
    "constraints": {"brier_must_not_increase": True, "brier_tolerance": 0.0,
                    "event_recall_at_budget_max_drop": 0.02},
    "reliability_bins": 10,
    "method_preference": ["identity", "temperature_scaling", "platt_scaling", "isotonic_regression"],
}


def test_pool_adjacent_violators_hand_example():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    thresholds, values = pool_adjacent_violators(x, y)
    assert thresholds.tolist() == x.tolist()
    assert np.all(np.diff(values) >= 0)
    # blocks: [1,0]->0.5 then [0.5,1,0]->0.5 ... final [x1..x4]=0.5, x5=1
    assert values.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5, 1.0])
    thresholds, values = pool_adjacent_violators(np.array([2.0, 2.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    assert thresholds.tolist() == [1.0, 2.0]
    assert values.tolist() == [0.0, 0.5]


def test_isotonic_matches_sklearn_when_available():
    isotonic = pytest.importorskip("sklearn.isotonic")
    rng = np.random.default_rng(11)
    scores = np.round(rng.random(500), 2)
    targets = (rng.random(500) < scores ** 3).astype(int)
    ours = fit_isotonic_regression(scores, targets)
    theirs = isotonic.IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, targets)
    grid = np.linspace(0.0, 1.0, 201)
    assert np.abs(ours.apply(grid) - theirs.predict(grid)).max() < 1e-12


def test_temperature_and_platt_reduce_nll_and_stay_monotone():
    rng = np.random.default_rng(2)
    logits = rng.normal(0.0, 3.0, 800)
    truth_probability = 1.0 / (1.0 + np.exp(-logits / 2.5))
    targets = (rng.random(800) < truth_probability).astype(int)
    scores = 1.0 / (1.0 + np.exp(-logits))
    temperature = fit_temperature_scaling(scores, targets)
    assert 1.8 < temperature.params["temperature"] < 3.5
    assert temperature.fit_summary["nll_after"] < temperature.fit_summary["nll_before"]
    platt = fit_platt_scaling(scores, targets)
    assert platt.fit_summary["nll_after"] <= platt.fit_summary["nll_before"]
    assert 0.25 < platt.params["slope"] < 0.6
    grid = np.linspace(0.0, 1.0, 50)
    for calibrator in (temperature, platt):
        assert np.all(np.diff(calibrator.apply(grid)) >= 0)
        assert calibrator.apply(np.array([0.0, 1.0])).tolist() == pytest.approx(
            calibrator.apply(np.array([0.0, 1.0])).tolist())


def test_serialisation_is_canonical_and_hash_stable():
    calibrator = Calibrator("platt_scaling", {"slope": 1.5, "intercept": -0.25})
    payload = calibrator.to_json_bytes()
    assert payload == calibrator.to_json_bytes()
    decoded = json.loads(payload)
    assert list(decoded) == sorted(decoded)
    restored = Calibrator.from_json_bytes(payload)
    assert restored == calibrator
    assert restored.sha256() == calibrator.sha256()
    with pytest.raises(ValueError, match="unknown calibration method"):
        Calibrator("spline")
    with pytest.raises(ValueError, match="finite"):
        Calibrator("temperature_scaling", {"temperature": float("nan")}).to_json_bytes()


def test_fit_calibrator_uses_eligible_validation_rows_only():
    rows = build_rows("p3_causal_tcn")
    calibrator = fit_calibrator("isotonic_regression", rows)
    assert calibrator.fit_summary["fit_rows"] == sum(
        1 for row in rows if row["eligibility"].startswith("eligible")
    )
    with pytest.raises(ValueError, match="validation rows only"):
        fit_calibrator("isotonic_regression", build_rows("p3_causal_tcn", split="development"))
    calibrated = apply_calibrator_rows(rows, calibrator)
    assert all(0.0 <= row["risk_score"] <= 1.0 for row in calibrated)
    assert all(row["raw_score"] == original["raw_score"] for row, original in zip(calibrated, rows))


def test_select_calibration_reports_before_after_and_respects_constraints(monkeypatch):
    rows = build_rows("p3_causal_tcn")

    def constant_fitter(scores, targets):
        rate = float(targets.mean())
        return Calibrator("platt_scaling", {"slope": 0.0, "intercept": float(np.log(rate / (1 - rate)))})

    monkeypatch.setitem(calibrators.FITTERS, "platt_scaling", constant_fitter)
    report = select_calibration(rows, policy=POLICY, alarm=ROOT_ALARM)
    assert report["selection_split"] == "validation"
    platt = next(c for c in report["candidates"] if c["method"] == "platt_scaling")
    assert platt["ece"] == pytest.approx(0.0, abs=1e-9)
    assert "event_recall_at_budget_dropped_beyond_tolerance" in platt["constraint_violations"]
    assert "brier_score_increased" in platt["constraint_violations"]
    assert report["chosen_method"] != "platt_scaling"
    chosen = report["after"]
    assert chosen["ece"] <= report["before"]["ece"]
    assert chosen["brier_score"] <= report["before"]["brier_score"]
    assert chosen["event_recall_at_budget"] >= report["before"]["event_recall_at_budget"] - 0.02
    assert report["before"]["event_recall_at_budget"] == 0.75
    assert len(chosen["reliability_bins"]) == 10
    assert report["calibrator"].sha256() == report["chosen_calibrator_sha256"]
    yaml.safe_dump({k: v for k, v in report.items() if k != "calibrator"})


def test_select_calibration_falls_back_to_identity_when_nothing_feasible():
    rows = build_rows("p3_causal_tcn")
    strict = {**POLICY, "constraints": {"brier_must_not_increase": True, "brier_tolerance": -1.0,
                                        "event_recall_at_budget_max_drop": 0.0}}
    report = select_calibration(rows, policy=strict, alarm=ROOT_ALARM)
    assert report["chosen_method"] == "identity"
    with pytest.raises(ValueError, match="validation split"):
        select_calibration(rows, policy={**POLICY, "selection_split": "development"}, alarm=ROOT_ALARM)

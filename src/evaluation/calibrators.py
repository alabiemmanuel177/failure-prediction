"""Validation-fitted probability calibrators with canonical, hash-addressed serialisation.

Every calibrator maps the uncalibrated ``raw_score`` in [0, 1] to a calibrated
``risk_score`` in [0, 1]. Fitting uses eligible validation decisions only; inference
depends on numpy alone so the deployed policy never needs scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .calibration import reliability_curve
from .policy import select_validation_threshold
from .prediction_tables import (
    ELIGIBLE, clean_run_ids, group_episodes, policy_settings, require_validation_only,
    with_risk_scores,
)


SCHEMA_VERSION = 1
EPSILON = 1e-6
METHODS = ("identity", "temperature_scaling", "platt_scaling", "isotonic_regression")


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _nll(probabilities: np.ndarray, targets: np.ndarray) -> float:
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return float(-np.mean(targets * np.log(clipped) + (1 - targets) * np.log1p(-clipped)))


def _validate_inputs(scores: np.ndarray, targets: np.ndarray) -> None:
    if scores.ndim != 1 or targets.shape != scores.shape:
        raise ValueError("scores and targets must be one-dimensional and aligned")
    if scores.size == 0:
        raise ValueError("no eligible calibration rows")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("raw scores must be finite and lie in [0, 1]")
    if not set(np.unique(targets)).issubset({0, 1}):
        raise ValueError("targets must be binary")


@dataclass(frozen=True)
class Calibrator:
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    fit_summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown calibration method: {self.method}")

    def apply(self, raw_scores: Sequence[float]) -> np.ndarray:
        scores = np.asarray(raw_scores, dtype=float)
        if scores.size and (np.any(scores < 0.0) or np.any(scores > 1.0)):
            raise ValueError("raw scores must lie in [0, 1]")
        if self.method == "identity":
            output = scores.copy()
        elif self.method == "temperature_scaling":
            output = _sigmoid(_logit(scores) / float(self.params["temperature"]))
        elif self.method == "platt_scaling":
            output = _sigmoid(
                float(self.params["slope"]) * _logit(scores) + float(self.params["intercept"])
            )
        else:
            output = np.interp(
                scores,
                np.asarray(self.params["thresholds"], dtype=float),
                np.asarray(self.params["values"], dtype=float),
            )
        return np.clip(output, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "method": self.method,
            "params": _plain(self.params),
            "fit_summary": _plain(self.fit_summary),
        }

    def to_json_bytes(self) -> bytes:
        return canonical_json(self.to_dict())

    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Calibrator":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported calibrator schema_version")
        return cls(
            method=str(payload["method"]),
            params=dict(payload.get("params", {})),
            fit_summary=dict(payload.get("fit_summary", {})),
        )

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> "Calibrator":
        return cls.from_dict(json.loads(payload.decode("utf-8")))


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _plain(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("calibrator parameters must be finite")
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def fit_temperature_scaling(scores: np.ndarray, targets: np.ndarray) -> Calibrator:
    """Golden-section search of log-temperature minimising validation NLL."""
    _validate_inputs(scores, targets)
    logits = _logit(scores)

    def objective(log_temperature: float) -> float:
        return _nll(_sigmoid(logits / math.exp(log_temperature)), targets)

    lower, upper = -5.0, 5.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    value_left, value_right = objective(left), objective(right)
    for _ in range(200):
        if value_left < value_right:
            upper, right, value_right = right, left, value_left
            left = upper - ratio * (upper - lower)
            value_left = objective(left)
        else:
            lower, left, value_left = left, right, value_right
            right = lower + ratio * (upper - lower)
            value_right = objective(right)
    log_temperature = (lower + upper) / 2.0
    temperature = math.exp(log_temperature)
    return Calibrator(
        "temperature_scaling",
        {"temperature": temperature},
        {
            "fit_rows": int(scores.size), "positive_rows": int(targets.sum()),
            "nll_before": _nll(scores, targets), "nll_after": objective(log_temperature),
        },
    )


def fit_platt_scaling(
    scores: np.ndarray, targets: np.ndarray, *, ridge: float = 1e-4, iterations: int = 100
) -> Calibrator:
    """Damped Newton fit of sigmoid(slope * logit + intercept) with a tiny ridge penalty."""
    _validate_inputs(scores, targets)
    logits = _logit(scores)
    design = np.column_stack([logits, np.ones_like(logits)])
    weights = np.array([1.0, 0.0])
    penalty = ridge * np.eye(2)

    def loss(candidate: np.ndarray) -> float:
        probabilities = _sigmoid(design @ candidate)
        return _nll(probabilities, targets) + 0.5 * ridge * float(candidate @ candidate)

    current = loss(weights)
    for _ in range(iterations):
        probabilities = _sigmoid(design @ weights)
        gradient = design.T @ (probabilities - targets) / scores.size + ridge * weights
        curvature = probabilities * (1.0 - probabilities)
        hessian = (design * curvature[:, None]).T @ design / scores.size + penalty
        step = np.linalg.solve(hessian, gradient)
        scale = 1.0
        improved = False
        while scale > 1e-6:
            candidate = weights - scale * step
            candidate_loss = loss(candidate)
            if candidate_loss <= current:
                weights, current, improved = candidate, candidate_loss, True
                break
            scale *= 0.5
        if not improved or float(np.max(np.abs(scale * step))) < 1e-10:
            break
    return Calibrator(
        "platt_scaling",
        {"slope": float(weights[0]), "intercept": float(weights[1])},
        {
            "fit_rows": int(scores.size), "positive_rows": int(targets.sum()),
            "ridge": ridge, "nll_before": _nll(scores, targets),
            "nll_after": _nll(_sigmoid(design @ weights), targets),
        },
    )


def pool_adjacent_violators(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted non-decreasing isotonic fit; returns unique x and fitted values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    if x.size == 0:
        raise ValueError("isotonic fit requires at least one point")
    order = np.argsort(x, kind="stable")
    x, y, weights = x[order], y[order], weights[order]
    unique_x, start = np.unique(x, return_index=True)
    bounds = list(start) + [x.size]
    block_weight = np.array([weights[a:b].sum() for a, b in zip(bounds, bounds[1:])])
    block_value = np.array([
        (y[a:b] * weights[a:b]).sum() / weights[a:b].sum() for a, b in zip(bounds, bounds[1:])
    ])
    values: list[float] = []
    sizes: list[float] = []
    counts: list[int] = []
    for value, weight in zip(block_value, block_weight):
        values.append(float(value))
        sizes.append(float(weight))
        counts.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            merged_weight = sizes[-2] + sizes[-1]
            merged_value = (values[-2] * sizes[-2] + values[-1] * sizes[-1]) / merged_weight
            values[-2:] = [merged_value]
            sizes[-2:] = [merged_weight]
            counts[-2:] = [counts[-2] + counts[-1]]
    fitted = np.repeat(np.array(values), np.array(counts))
    return unique_x, fitted


def fit_isotonic_regression(scores: np.ndarray, targets: np.ndarray) -> Calibrator:
    _validate_inputs(scores, targets)
    thresholds, values = pool_adjacent_violators(scores, targets.astype(float))
    values = np.clip(values, 0.0, 1.0)
    calibrated = np.interp(scores, thresholds, values)
    return Calibrator(
        "isotonic_regression",
        {"thresholds": thresholds.tolist(), "values": values.tolist()},
        {
            "fit_rows": int(scores.size), "positive_rows": int(targets.sum()),
            "block_count": int(np.unique(values).size),
            "nll_before": _nll(scores, targets), "nll_after": _nll(calibrated, targets),
        },
    )


def identity_calibrator(scores: np.ndarray, targets: np.ndarray) -> Calibrator:
    _validate_inputs(scores, targets)
    return Calibrator(
        "identity", {},
        {"fit_rows": int(scores.size), "positive_rows": int(targets.sum()),
         "nll_before": _nll(scores, targets), "nll_after": _nll(scores, targets)},
    )


FITTERS = {
    "identity": identity_calibrator,
    "temperature_scaling": fit_temperature_scaling,
    "platt_scaling": fit_platt_scaling,
    "isotonic_regression": fit_isotonic_regression,
}


def eligible_arrays(rows: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    scores, targets = [], []
    for row in rows:
        if row.get("eligibility") not in ELIGIBLE:
            continue
        scores.append(float(row["raw_score"]))
        targets.append(1 if row.get("eligibility") == "eligible_positive" else 0)
    return np.asarray(scores, dtype=float), np.asarray(targets, dtype=int)


def fit_calibrator(method: str, rows: Sequence[Mapping[str, object]]) -> Calibrator:
    """Fit ``method`` on eligible validation decisions of a prediction table."""
    if method not in FITTERS:
        raise ValueError(f"unknown calibration method: {method}")
    require_validation_only(rows, "calibration fitting")
    scores, targets = eligible_arrays(rows)
    return FITTERS[method](scores, targets)


def apply_calibrator_rows(
    rows: Sequence[Mapping[str, object]], calibrator: Calibrator
) -> list[dict[str, object]]:
    raw = np.asarray([float(row["raw_score"]) for row in rows], dtype=float)
    return with_risk_scores(rows, calibrator.apply(raw).tolist())


def evaluate_calibrated_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    alarm: Mapping[str, object],
    bins: int,
) -> dict[str, object]:
    """Reliability, Brier and event recall at the frozen budget for calibrated rows."""
    curve = reliability_curve(rows, bins=bins)
    settings = policy_settings(alarm)
    episodes = group_episodes(rows)
    clean = clean_run_ids(rows)
    try:
        selection = select_validation_threshold(
            episodes, clean,
            false_alert_budget=settings["false_alert_budget"],
            required_above=settings["required_above"],
            decisions_considered=settings["decisions_considered"],
            cooldown_seconds=settings["cooldown_seconds"],
        )
        recall_block = {
            "threshold_at_budget": selection["threshold"],
            "event_recall_at_budget": selection["event_recall"],
            "false_alerts_per_clean_mission_at_budget": selection["false_alerts_per_clean_mission"],
            "median_useful_lead_seconds_at_budget": selection["median_useful_lead_seconds_detected"],
        }
    except ValueError as error:
        recall_block = {
            "threshold_at_budget": None, "event_recall_at_budget": None,
            "false_alerts_per_clean_mission_at_budget": None,
            "median_useful_lead_seconds_at_budget": None,
            "threshold_selection_error": str(error),
        }
    return {
        "brier_score": curve["brier_score"],
        "ece": curve["ece"],
        "eligible_decision_count": curve["eligible_decision_count"],
        "reliability_bins": curve["bins"],
        **recall_block,
    }


def select_calibration(
    rows: Sequence[Mapping[str, object]],
    *,
    policy: Mapping[str, object],
    alarm: Mapping[str, object],
) -> dict[str, object]:
    """Pick the method minimising validation ECE under the H3 constraints.

    ``identity`` (raw scores) is the reference and always feasible, so "no
    calibration" is a legitimate outcome when every learned method degrades Brier or
    recall beyond the declared tolerance.
    """
    if policy.get("selection_split") != "validation":
        raise ValueError("calibration policy must select on the validation split")
    require_validation_only(rows, "calibration selection")
    methods = list(policy.get("candidate_methods", METHODS))
    if "identity" not in methods:
        methods.insert(0, "identity")
    unknown = [method for method in methods if method not in FITTERS]
    if unknown:
        raise ValueError(f"unknown calibration methods in policy: {unknown}")
    constraints = policy.get("constraints", {})
    brier_guard = bool(constraints.get("brier_must_not_increase", True))
    brier_tolerance = float(constraints.get("brier_tolerance", 0.0))
    recall_tolerance = float(constraints.get("event_recall_at_budget_max_drop", 0.02))
    bins = int(policy.get("reliability_bins", 10))
    preference = list(policy.get("method_preference", METHODS))

    fitted: dict[str, Calibrator] = {}
    candidates: dict[str, dict[str, object]] = {}
    for method in methods:
        calibrator = fit_calibrator(method, rows)
        calibrated = apply_calibrator_rows(rows, calibrator)
        fitted[method] = calibrator
        candidates[method] = {
            "method": method,
            "calibrator_sha256": calibrator.sha256(),
            "params": calibrator.to_dict()["params"] if method != "isotonic_regression" else {
                "block_count": calibrator.fit_summary.get("block_count"),
            },
            "fit_summary": calibrator.to_dict()["fit_summary"],
            **evaluate_calibrated_rows(calibrated, alarm=alarm, bins=bins),
        }

    reference = candidates["identity"]
    reference_recall = reference["event_recall_at_budget"]
    for method, candidate in candidates.items():
        reasons = []
        if method == "identity":  # the raw reference is feasible by definition
            candidate["constraint_violations"] = reasons
            candidate["feasible"] = True
            continue
        if brier_guard and candidate["brier_score"] > reference["brier_score"] + brier_tolerance:
            reasons.append("brier_score_increased")
        recall = candidate["event_recall_at_budget"]
        if reference_recall is not None:
            if recall is None or recall < reference_recall - recall_tolerance:
                reasons.append("event_recall_at_budget_dropped_beyond_tolerance")
        candidate["constraint_violations"] = reasons
        candidate["feasible"] = not reasons

    def ranking(method: str) -> tuple[float, float, float, int]:
        candidate = candidates[method]
        recall = candidate["event_recall_at_budget"]
        return (
            float(candidate["ece"]),
            float(candidate["brier_score"]),
            -(float(recall) if recall is not None else -1.0),
            preference.index(method) if method in preference else len(preference),
        )

    feasible = [method for method in methods if candidates[method]["feasible"]]
    chosen = min(feasible, key=ranking)
    return {
        "selection_split": "validation",
        "objective": policy.get("objective", "minimum_validation_ece"),
        "constraints": {
            "brier_must_not_increase": brier_guard,
            "brier_tolerance": brier_tolerance,
            "event_recall_at_budget_max_drop": recall_tolerance,
        },
        "reliability_bins": bins,
        "candidate_methods": methods,
        "chosen_method": chosen,
        "chosen_calibrator_sha256": fitted[chosen].sha256(),
        "before": reference,
        "after": candidates[chosen],
        "candidates": [candidates[method] for method in methods],
        "calibrator": fitted[chosen],
    }

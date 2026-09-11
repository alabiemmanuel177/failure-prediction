"""Cost-sensitive R3 selector fitted only on guard-eligible replay/simulation rows.

The model is a per-action ridge regression from warning-state features to observed
cost, implemented with numpy only so that it runs under the system interpreter.
Actions that the frozen guard rejects are never training targets (Protocol section
10, ``selector_training_rule``); the trainer asserts this against
``src.recovery.guards.eligible_actions`` rather than trusting the table.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .guards import ACTIONS, GuardConfig, RobotState, eligible_actions
from .selector import SIGNAL_ACTIONS


MODEL_ID = "r3_cost_sensitive_ridge_v1"
SIGNAL_GROUPS = tuple(sorted(SIGNAL_ACTIONS))
STATE_BOOLEANS = (
    "stopped", "stop_allowed", "localisation_poor", "planning_stale_or_blocked",
    "immediate_collision_risk", "obstruction_may_be_transient", "relocalisation_available",
)
STATE_CLEARANCES = ("rear_clearance_m", "rotation_clearance_m")
FEATURE_NAMES: tuple[str, ...] = (
    "risk_score",
    *(f"signal_group={group}" for group in SIGNAL_GROUPS),
    *STATE_BOOLEANS,
    *(name for clearance in STATE_CLEARANCES for name in (clearance, f"{clearance}_missing")),
    "repeated_recovery_count",
)
REQUIRED_COLUMNS = (
    "split", "warning_id", "risk_score", "diagnosed_signal_group",
    *STATE_BOOLEANS, *STATE_CLEARANCES, "repeated_recovery_count",
    "candidate_action", "guard_eligible", "observed_cost",
)
FORBIDDEN_TRAINING_SPLITS = ("held_out_map_test", "test")
ALLOWED_FIT_SPLITS = ("development", "validation")


def _truth(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"expected a boolean, got {value!r}")
    return text in {"true", "1", "yes"}


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return None if text in {"", "none", "null", "nan"} else float(text)


def state_from_row(row: Mapping[str, object]) -> RobotState:
    return RobotState(
        stopped=_truth(row["stopped"]), stop_allowed=_truth(row["stop_allowed"]),
        localisation_poor=_truth(row["localisation_poor"]),
        planning_stale_or_blocked=_truth(row["planning_stale_or_blocked"]),
        rear_clearance_m=_optional_float(row["rear_clearance_m"]),
        rotation_clearance_m=_optional_float(row["rotation_clearance_m"]),
        immediate_collision_risk=_truth(row["immediate_collision_risk"]),
        obstruction_may_be_transient=_truth(row["obstruction_may_be_transient"]),
        relocalisation_available=_truth(row.get("relocalisation_available", False)),
        repeated_recovery_count=int(row["repeated_recovery_count"]),
    )


def features_from_state(risk_score: float, signal_group: str, state: RobotState) -> list[float]:
    if not 0.0 <= float(risk_score) <= 1.0:
        raise ValueError("risk_score must lie in [0, 1]")
    group = signal_group if signal_group in SIGNAL_GROUPS else "unknown"
    features = [float(risk_score)]
    features.extend(1.0 if group == candidate else 0.0 for candidate in SIGNAL_GROUPS)
    features.extend(float(getattr(state, name)) for name in STATE_BOOLEANS)
    for clearance in STATE_CLEARANCES:
        value = getattr(state, clearance)
        features.extend([0.0 if value is None else float(value), 1.0 if value is None else 0.0])
    features.append(float(state.repeated_recovery_count))
    assert len(features) == len(FEATURE_NAMES)
    return features


def features_from_row(row: Mapping[str, object]) -> list[float]:
    return features_from_state(
        float(row["risk_score"]), str(row["diagnosed_signal_group"]), state_from_row(row)
    )


def read_cost_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"cost table lacks required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("cost table is empty")
    return rows


def _fit_ridge(matrix: np.ndarray, target: np.ndarray, ridge_lambda: float) -> tuple[float, np.ndarray]:
    design = np.hstack([np.ones((matrix.shape[0], 1)), matrix])
    penalty = ridge_lambda * np.eye(design.shape[1])
    penalty[0, 0] = 0.0  # the intercept is not penalised
    solution = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return float(solution[0]), solution[1:]


def train_selector(
    rows: Sequence[Mapping[str, object]],
    guard_config: GuardConfig,
    *,
    ridge_lambda: float = 1.0,
    minimum_rows_per_action: int = 5,
    seed: int = 20260903,
    fit_split: str = "development",
    fit_split_admitted_by: str | None = None,
) -> dict[str, Any]:
    """Fit per-action cost models from guard-eligible rows of one fitting split only.

    ``fit_split`` is ``development`` by default. ``validation`` is admitted only with
    the protocol amendment id that authorises it (PA-2026-09-03-04 trains the R3
    selector on the post-freeze validation-map recovery pilot). Held-out rows are
    refused unconditionally.
    """
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must be nonnegative")
    guard_config.validate()
    if fit_split in FORBIDDEN_TRAINING_SPLITS:
        raise ValueError("refusing to fit the selector on held-out rows")
    if fit_split not in ALLOWED_FIT_SPLITS:
        raise ValueError(f"fit_split must be one of {ALLOWED_FIT_SPLITS}, got {fit_split!r}")
    if fit_split == "validation" and not fit_split_admitted_by:
        raise ValueError(
            "fitting on validation rows requires the admitting protocol amendment id"
        )
    splits = {str(row["split"]) for row in rows}
    forbidden = splits & set(FORBIDDEN_TRAINING_SPLITS)
    if forbidden:
        raise ValueError(f"refusing to train the selector on protected rows: {sorted(forbidden)}")
    development = [row for row in rows if str(row["split"]) == fit_split]
    if not development:
        raise ValueError(f"selector fitting requires {fit_split}-split rows")
    training: dict[str, list[tuple[list[float], float]]] = {action: [] for action in ACTIONS}
    rejected_by_guard = 0
    mismatches = 0
    for row in development:
        action = str(row["candidate_action"])
        if action not in ACTIONS:
            raise ValueError(f"unknown candidate action: {action}")
        state = state_from_row(row)
        recomputed = eligible_actions(state, guard_config)[action][0]
        declared = _truth(row["guard_eligible"])
        if declared != recomputed:
            mismatches += 1
            continue
        if not recomputed:
            rejected_by_guard += 1
            continue
        training[action].append((features_from_row(row), float(row["observed_cost"])))
    if mismatches:
        raise AssertionError(
            f"{mismatches} rows declare guard eligibility that disagrees with the frozen guard"
        )
    assert all(
        eligible_actions(state_from_row(row), guard_config)[str(row["candidate_action"])][0]
        for row in development if _truth(row["guard_eligible"])
    ), "a guard-rejected action reached the training set"
    all_costs = [cost for samples in training.values() for _f, cost in samples]
    if not all_costs:
        raise ValueError("no guard-eligible development rows to fit")
    unfitted_cost = float(max(all_costs))
    actions: dict[str, Any] = {}
    unfitted: list[str] = []
    for action in ACTIONS:
        samples = training[action]
        if len(samples) < minimum_rows_per_action:
            unfitted.append(action)
            continue
        matrix = np.asarray([features for features, _c in samples], dtype=float)
        target = np.asarray([cost for _f, cost in samples], dtype=float)
        intercept, weights = _fit_ridge(matrix, target, ridge_lambda)
        predicted = intercept + matrix @ weights
        actions[action] = {
            "intercept": intercept,
            "weights": [float(value) for value in weights],
            "training_rows": len(samples),
            "mean_observed_cost": float(target.mean()),
            "training_rmse": float(np.sqrt(np.mean((predicted - target) ** 2))),
        }
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "policy_id": "R3",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_names": list(FEATURE_NAMES),
        "actions": actions,
        "unfitted_actions": unfitted,
        "unfitted_action_cost": unfitted_cost,
        "ridge_lambda": ridge_lambda,
        "minimum_rows_per_action": minimum_rows_per_action,
        "seed": seed,
        "training_splits_used": [fit_split],
        "fit_split": fit_split,
        "fit_split_admitted_by": fit_split_admitted_by if fit_split != "development" else None,
        "row_counts": {
            "input": len(rows),
            "development": len(development),
            "guard_rejected_excluded": rejected_by_guard,
            "fitted": int(sum(len(samples) for samples in training.values())),
        },
        "selector_training_rule": "actions_rejected_by_guard_are_never_training_targets",
        "protected_test_used": False,
    }


def predict_action_costs(model: Mapping[str, Any], features: Sequence[float]) -> dict[str, float]:
    if list(model["feature_names"]) != list(FEATURE_NAMES):
        raise ValueError("selector model feature contract differs from this code")
    vector = np.asarray(features, dtype=float)
    costs = {}
    for action in ACTIONS:
        fitted = model["actions"].get(action)
        if fitted is None:
            costs[action] = float(model["unfitted_action_cost"])
        else:
            costs[action] = float(fitted["intercept"] + vector @ np.asarray(fitted["weights"]))
    return costs


def predict_costs_for_request(
    model: Mapping[str, Any], risk_score: float, signal_group: str, state: RobotState
) -> dict[str, float]:
    return predict_action_costs(model, features_from_state(risk_score, signal_group, state))


def evaluate_selector_regret(
    model: Mapping[str, Any], rows: Iterable[Mapping[str, object]], guard_config: GuardConfig
) -> dict[str, Any]:
    """Regret of the predicted-lowest eligible action versus the observed-lowest one."""
    by_warning: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_warning.setdefault(str(row["warning_id"]), []).append(row)
    regrets = []
    matches = 0
    for warning_id, candidates in sorted(by_warning.items()):
        observed = {}
        for row in candidates:
            state = state_from_row(row)
            if eligible_actions(state, guard_config)[str(row["candidate_action"])][0]:
                observed[str(row["candidate_action"])] = float(row["observed_cost"])
        if not observed:
            continue
        first = candidates[0]
        predicted = predict_action_costs(model, features_from_row(first))
        chosen = min(
            ((action, predicted[action]) for action in observed), key=lambda item: (item[1], item[0])
        )[0]
        best_cost = min(observed.values())
        regrets.append(observed[chosen] - best_cost)
        matches += int(observed[chosen] == best_cost)
    return {
        "warning_count": len(regrets),
        "mean_regret": float(np.mean(regrets)) if regrets else None,
        "oracle_match_rate": matches / len(regrets) if regrets else None,
    }


def model_bytes(model: Mapping[str, Any]) -> bytes:
    return (json.dumps(model, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def save_selector_model(model: Mapping[str, Any], path: Path) -> str:
    """Write the JSON model and a ``.sha256`` sidecar; refuse to overwrite either."""
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite selector model: {path}")
    payload = model_bytes(model)
    digest = sha256_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    with sidecar.open("x", encoding="utf-8") as stream:
        stream.write(f"{digest}  {path.name}\n")
    return digest


def load_selector_model(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        raise FileNotFoundError(f"selector model has no sha256 sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if sha256_bytes(payload) != expected:
        raise ValueError(f"selector model checksum mismatch: {path}")
    model = json.loads(payload.decode("utf-8"))
    if model.get("model_id") != MODEL_ID:
        raise ValueError(f"unexpected selector model id: {model.get('model_id')}")
    return model

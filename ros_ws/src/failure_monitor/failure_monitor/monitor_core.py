"""Online failure-monitor core: live 2 Hz windows through the offline feature code.

No ROS import lives here so the replay-equivalence test can drive the buffer with a
recorded telemetry CSV under the system interpreter. Every transformation reuses the
offline pipeline verbatim:

* raw scalars       ``scripts.extract_bag_scalar_telemetry.feature_values`` (per topic)
* causal resampling ``src.features.extract_decision_rows`` (newest non-future sample)
* temporal features ``src.features.derive_window_features``
* column order      ``src.features.model_columns(primary)``
* mask + z-score    ``scripts.predict_decisions.LoadedModel.prepare`` (development bundle)
* model             ``LoadedModel.score_prepared`` (same classes as ``predict_decisions``)
* calibration       ``src.evaluation.calibrators.Calibrator``
* alarm rule        ``src.evaluation.policy.apply_alarm_policy`` replayed over the stream

Decision grid. Offline, ``assemble_causal_sequences`` samples ``start + k * stride`` for
``k >= 1`` and the first labelled decision is the first grid point with a complete
``history_seconds`` window, so ``decision_index = k - steps``. The live grid anchors
``start`` on mission start (first ``/behavior_tree_log`` message, the goal-dispatch
signal the fault injector also uses); the offline grid anchors on the label-only
``goal_dispatched`` event that the runner publishes a few milliseconds earlier. Offline
timestamps are recorder receive times; live timestamps are the monitor's receive
times on the same simulation clock.

Exact incremental enrichment. ``derive_window_features`` is a whole-sequence function
whose only state older than the five-second history is ``previous_amcl`` (for
``pose_jump``). Running it on the suffix that starts at the earlier of the last
AMCL-observed row and ``k - steps`` therefore reproduces row ``k`` exactly; the
unit test asserts equality with the full recomputation.

Diagnosed signal group. After mask and normalisation, the last time step's value
channels are grouped by ``configs/ablations.yaml`` ``feature_groups``; a group's
contribution is the mean |z| over its observed (missing = 0) features and the group
with the largest contribution is reported. Groups map onto the recovery selector's
signal vocabulary: localisation -> localisation, planner_controller -> planning,
raw_sensor -> frontal_blockage, perception_confidence -> perception,
motion_and_goal -> unknown (controlled stop: motion anomalies alone do not identify
a transient obstruction). Ties or no observed feature -> unknown.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(os.environ.get("RESEARCH2_ROOT", Path(__file__).resolve().parents[4]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.evaluation.calibrators import Calibrator  # noqa: E402
from src.evaluation.policy import AlarmPolicy, apply_alarm_policy  # noqa: E402
from src.features import (  # noqa: E402
    LeakagePolicy, ScalarSample, derive_window_features, extract_decision_rows,
    load_primary_feature_set, load_raw_feature_contract, model_columns, resolve_feature_specs,
)
from src.recovery.guards import RobotState  # noqa: E402
from src.recovery.plumbing import forced_action, validate_policy_id  # noqa: E402
from extract_bag_scalar_telemetry import MAX_AGE, feature_values  # noqa: E402


WARNING_TOPIC = "/research2/warning"
EVENT_TOPIC = "/research2/events"
REQUEST_TOPIC = "/research2/recovery_requests"
MISSION_START_TOPIC = "/behavior_tree_log"
# Deployable topics the monitor may subscribe to. ``feature_values`` turns each message
# into raw contract scalars; the perception summary is the canonical compact_v2 source
# so the legacy ``/semantic/*`` images are never mixed in.
FEATURE_TOPICS = (
    "/cmd_vel", "/odom", "/amcl_pose", "/scan", "/plan", "/local_plan",
    "/research2/features/perception",
)
SUBSCRIBED_TOPICS = (*FEATURE_TOPICS, MISSION_START_TOPIC)
GROUP_TO_SIGNAL = {
    "localisation": "localisation",
    "planner_controller": "planning",
    "raw_sensor": "frontal_blockage",
    "perception_confidence": "perception",
    "motion_and_goal": "unknown",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MonitorRefused(RuntimeError):
    """Raised when the monitor must not start (freeze, leakage, assets)."""


# ----------------------------------------------------------------------------- assets


@dataclass(frozen=True)
class MonitorAssets:
    model_dir: Path
    calibrator_path: Path
    checkpoint_sha256: str
    calibrator_sha256: str
    alarm_policy: AlarmPolicy
    threshold_source: str
    engineering_smoke: bool
    freeze_path: Path | None
    freeze_sha256: str | None
    frozen: bool

    def record(self) -> dict[str, Any]:
        return {
            "model_dir": str(self.model_dir),
            "calibrator": str(self.calibrator_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "calibrator_sha256": self.calibrator_sha256,
            "threshold": self.alarm_policy.threshold,
            "persistence": [self.alarm_policy.required_above, self.alarm_policy.decisions_considered],
            "cooldown_seconds": self.alarm_policy.cooldown_seconds,
            "threshold_source": self.threshold_source,
            "engineering_smoke": self.engineering_smoke,
            "model_freeze": str(self.freeze_path) if self.freeze_path else None,
            "model_freeze_sha256": self.freeze_sha256,
            "frozen": self.frozen,
        }


def load_alarm_policy(path: Path, *, smoke_threshold: float | None) -> tuple[AlarmPolicy, str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    threshold = document.get("threshold")
    if threshold is None:
        if smoke_threshold is None:
            raise MonitorRefused(f"{path} has no frozen threshold; refusing to start")
        threshold, source = float(smoke_threshold), "engineering_smoke_parameter"
    else:
        threshold, source = float(threshold), str(path)
    policy = AlarmPolicy(
        threshold,
        int(document["persistence"]["required_above_threshold"]),
        int(document["persistence"]["decisions_considered"]),
        float(document["cooldown_seconds"]),
    )
    policy.validate()
    return policy, source


def resolve_monitor_assets(
    root: Path, *, freeze_path: Path, alarm_policy_path: Path, smoke_unfrozen: bool,
    model_dir_override: str = "", calibrator_override: str = "",
    smoke_threshold: float | None = None,
) -> MonitorAssets:
    """Fail closed: a frozen model or an explicit engineering smoke run, nothing else."""
    freeze = None
    if freeze_path.exists():
        freeze = yaml.safe_load(freeze_path.read_text(encoding="utf-8")) or {}
    frozen = bool(isinstance(freeze, dict) and freeze.get("frozen") is True)
    if not frozen and not smoke_unfrozen:
        raise MonitorRefused(
            f"{freeze_path} is absent or not frozen; the online monitor refuses to start "
            "(pass smoke_unfrozen:=true for an engineering-only smoke run)"
        )
    if frozen:
        predictor = freeze.get("predictor", {})
        calibration = freeze.get("calibration", {})
        checkpoint = root / str(predictor.get("checkpoint", ""))
        model_dir = checkpoint.parent
        calibrator_path = root / str(calibration.get("artifact", ""))
        if not checkpoint.is_file() or not calibrator_path.is_file():
            raise MonitorRefused("frozen predictor checkpoint or calibrator is missing")
        checkpoint_sha = sha256_file(checkpoint)
        calibrator_sha = sha256_file(calibrator_path)
        if checkpoint_sha != predictor.get("checkpoint_sha256"):
            raise MonitorRefused("checkpoint.pt differs from the frozen checkpoint_sha256")
        if calibrator_sha != calibration.get("artifact_sha256"):
            raise MonitorRefused("calibrator differs from the frozen artifact_sha256")
        if (model_dir_override and Path(model_dir_override).resolve() != model_dir.resolve()) or (
            calibrator_override and Path(calibrator_override).resolve() != calibrator_path.resolve()
        ):
            raise MonitorRefused("overrides are refused once the model is frozen")
        policy, source = load_alarm_policy(
            alarm_policy_path, smoke_threshold=smoke_threshold if smoke_unfrozen else None,
        )
    else:
        if not model_dir_override or not calibrator_override:
            raise MonitorRefused("smoke_unfrozen requires explicit model_dir and calibrator")
        model_dir = Path(model_dir_override)
        calibrator_path = Path(calibrator_override)
        checkpoint = model_dir / "checkpoint.pt"
        if not checkpoint.is_file() or not calibrator_path.is_file():
            raise MonitorRefused("smoke model_dir/checkpoint.pt or calibrator is missing")
        checkpoint_sha = sha256_file(checkpoint)
        calibrator_sha = sha256_file(calibrator_path)
        policy, source = load_alarm_policy(alarm_policy_path, smoke_threshold=smoke_threshold)
    return MonitorAssets(
        model_dir=model_dir, calibrator_path=calibrator_path,
        checkpoint_sha256=checkpoint_sha, calibrator_sha256=calibrator_sha,
        alarm_policy=policy, threshold_source=source,
        engineering_smoke=bool(smoke_unfrozen),
        freeze_path=freeze_path if freeze_path.exists() else None,
        freeze_sha256=sha256_file(freeze_path) if freeze_path.exists() else None,
        frozen=frozen,
    )


def check_subscriptions(policy: LeakagePolicy, topics: Sequence[str] = SUBSCRIBED_TOPICS) -> None:
    """Refuse any live subscription that the leakage denylist forbids."""
    for topic in topics:
        policy.validate(feature_name="live_subscription", source=topic)


# ----------------------------------------------------------------------------- buffer


class SampleBuffer:
    """Per-feature monotonic sample lists fed by live receive-time scalars."""

    def __init__(self, contract: Mapping[str, Mapping[str, Any]]):
        self.contract = contract
        self.samples: dict[str, list[ScalarSample]] = {}
        self.sources: dict[str, str] = {}
        self.dropped_out_of_order = 0
        self.dropped_unknown = 0
        self.dropped_mixed_source = 0
        self.received = 0

    def add(self, feature: str, source: str, timestamp: float, value: float) -> bool:
        if feature not in self.contract:
            self.dropped_unknown += 1
            return False
        if source not in self.contract[feature]["sources"]:
            self.dropped_unknown += 1
            return False
        known = self.sources.setdefault(feature, source)
        if known != source:
            self.dropped_mixed_source += 1
            return False
        if not (math.isfinite(timestamp) and math.isfinite(value)):
            self.dropped_unknown += 1
            return False
        series = self.samples.setdefault(feature, [])
        if series and timestamp < series[-1].timestamp:
            self.dropped_out_of_order += 1
            return False
        series.append(ScalarSample(float(timestamp), float(value)))
        self.received += 1
        return True

    def add_message(self, topic: str, message: Any, timestamp: float) -> int:
        """Convert one live message with the offline ``feature_values`` and buffer it."""
        added = 0
        for feature, value in feature_values(topic, message).items():
            added += int(self.add(feature, topic, timestamp, value))
        return added

    def contract_rows(self) -> list[dict[str, str]]:
        return [
            {"feature": name, "source": source,
             "max_age_seconds": str(self.contract[name]["max_age_seconds"])}
            for name, source in sorted(self.sources.items())
        ]

    def prune(self, decision_time: float) -> None:
        """Drop samples no future decision can select (older than max_age before now)."""
        for name, series in self.samples.items():
            cutoff = decision_time - float(self.contract[name]["max_age_seconds"])
            keep = 0
            while keep < len(series) - 1 and series[keep + 1].timestamp <= cutoff:
                keep += 1
            if keep:
                del series[:keep]

    def counters(self) -> dict[str, int]:
        return {
            "received": self.received,
            "dropped_out_of_order": self.dropped_out_of_order,
            "dropped_unknown": self.dropped_unknown,
            "dropped_mixed_source": self.dropped_mixed_source,
        }


# ----------------------------------------------------------------------------- windows


def grid_time(start: float, stride: float, index: int) -> float:
    """``k``-th grid point exactly as ``src.features.sequences._grid`` computes it."""
    current = Decimal(str(start))
    step = Decimal(str(stride))
    for _ in range(index):
        current += step
    return float(current)


class WindowBuilder:
    """Causal decision rows and complete windows from a live sample buffer."""

    def __init__(
        self, *, run_id: str, contract: Mapping[str, Mapping[str, Any]],
        primary_features: Sequence[str], leakage_policy: LeakagePolicy,
        goal_x: float, goal_y: float, history_seconds: float = 5.0,
        stride_seconds: float = 0.5,
    ):
        ratio = history_seconds / stride_seconds
        self.steps = round(ratio)
        if history_seconds <= 0 or stride_seconds <= 0 or not math.isclose(ratio, self.steps):
            raise ValueError("history must be a positive integer multiple of stride")
        self.run_id = run_id
        self.contract = contract
        self.columns = model_columns(primary_features)
        self.policy = leakage_policy
        self.goal = (float(goal_x), float(goal_y))
        self.history_seconds = float(history_seconds)
        self.stride = float(stride_seconds)
        self.buffer = SampleBuffer(contract)
        self.grid_start: float | None = None
        self.next_grid_index = 1
        self.raw_rows: list[dict[str, object]] = []
        self.raw_base = self.next_grid_index   # grid index of raw_rows[0] (grid starts at 1)
        self.last_amcl_grid_index: int | None = None
        self.enriched_tail: deque[dict[str, object]] = deque(maxlen=self.steps)

    def start(self, grid_start: float) -> None:
        if self.grid_start is not None:
            return
        self.grid_start = float(grid_start)

    def due_grid_times(self, now: float) -> list[tuple[int, float]]:
        if self.grid_start is None:
            return []
        due = []
        index = self.next_grid_index
        while True:
            moment = grid_time(self.grid_start, self.stride, index)
            if moment > now:
                break
            due.append((index, moment))
            index += 1
        return due

    def _raw_row(self, grid_index: int, decision_time: float) -> dict[str, object]:
        specs, _grouped = resolve_feature_specs(self.contract, self.buffer.contract_rows())
        row = extract_decision_rows(
            run_id=self.run_id, decision_times=[decision_time], specs=specs,
            samples_by_feature=self.buffer.samples, leakage_policy=self.policy,
        )[0]
        for name, value in row.items():
            if name.startswith("__audit_source_time__") and value is not None \
                    and float(value) > decision_time:
                raise AssertionError(f"future source timestamp in {name}")
        row["decision_index"] = grid_index - 1
        return row

    def _enrich_last(self) -> dict[str, object]:
        newest = self.raw_base + len(self.raw_rows) - 1
        suffix_start = newest - self.steps
        if self.last_amcl_grid_index is not None:
            suffix_start = min(suffix_start, self.last_amcl_grid_index)
        suffix_start = max(suffix_start, self.raw_base)
        rows = self.raw_rows[suffix_start - self.raw_base:]
        enriched = derive_window_features(
            rows, goal_x=self.goal[0], goal_y=self.goal[1], history_seconds=self.history_seconds,
        )
        # Rows older than the suffix start are never needed again.
        if suffix_start > self.raw_base:
            del self.raw_rows[: suffix_start - self.raw_base]
            self.raw_base = suffix_start
        return enriched[-1]

    def advance(self, grid_index: int, decision_time: float) -> np.ndarray | None:
        """Append grid point ``grid_index``; return the complete window or None."""
        if grid_index != self.next_grid_index:
            raise ValueError("grid points must be advanced consecutively")
        raw = self._raw_row(grid_index, decision_time)
        self.raw_rows.append(raw)
        # The suffix must start no later than the last AMCL-observed row *before* this
        # one, because pose_jump of this row is the distance from that earlier fix.
        enriched = self._enrich_last()
        if int(raw["amcl_x__missing"]) == 0 and int(raw["amcl_y__missing"]) == 0:
            self.last_amcl_grid_index = grid_index
        self.enriched_tail.append(enriched)
        self.next_grid_index = grid_index + 1
        self.buffer.prune(decision_time)
        if len(self.enriched_tail) < self.steps:
            return None
        window = list(self.enriched_tail)
        lower = decision_time - self.history_seconds
        if not all(lower < float(row["decision_time"]) <= decision_time for row in window):
            raise AssertionError("window violates its causal history interval")
        return np.asarray(
            [[float(row[column]) for column in self.columns] for row in window], dtype=np.float32,
        )

    def decision_index(self, grid_index: int) -> int:
        return grid_index - self.steps


# ----------------------------------------------------------------------------- scoring


def load_feature_groups(path: Path) -> dict[str, list[str]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {str(name): list(members) for name, members in document["feature_groups"].items()}


def diagnose_signal_group(
    prepared_last_step: np.ndarray, feature_names: Sequence[str],
    feature_groups: Mapping[str, Sequence[str]],
) -> tuple[str, dict[str, float]]:
    """Largest mean |z| over observed value channels of the last time step (see module doc)."""
    index = {name: position for position, name in enumerate(feature_names)}
    contributions: dict[str, float] = {}
    for group, members in feature_groups.items():
        scores = []
        for feature in members:
            if feature not in index:
                continue
            if float(prepared_last_step[index[f"{feature}__missing"]]) > 0.5:
                continue
            scores.append(abs(float(prepared_last_step[index[feature]])))
        contributions[group] = float(np.mean(scores)) if scores else 0.0
    if not contributions or max(contributions.values()) <= 0.0:
        return "unknown", contributions
    ordered = sorted(contributions.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered) > 1 and math.isclose(ordered[0][1], ordered[1][1]):
        return "unknown", contributions
    return GROUP_TO_SIGNAL.get(ordered[0][0], "unknown"), contributions


@dataclass(frozen=True)
class StateEstimatorConfig:
    """Thresholds behind the guard inputs the monitor reports (engineering estimates)."""

    stopped_linear_mps: float = 0.01
    stopped_angular_rads: float = 0.05
    localisation_poor_covariance_trace: float = 0.5
    planning_stale_replan_rate_per_s: float = 1.0
    immediate_collision_front_range_m: float = 0.30
    transient_obstruction_front_range_m: float = 1.0
    scan_max_age_seconds: float = 0.5
    odom_max_age_seconds: float = 0.5


def estimate_robot_state(
    window_row: Mapping[str, object], *, diagnosed: str, aux: Mapping[str, tuple[float, float]],
    now: float, config: StateEstimatorConfig, relocalisation_available: bool,
    repeated_recovery_count: int,
) -> RobotState:
    """Guard inputs from the last enriched row plus rear/rotation clearances.

    ``aux`` carries ``(timestamp, value)`` for ``rear_clearance_m`` and
    ``rotation_clearance_m`` computed from the newest scan; a clearance older than
    ``scan_max_age_seconds`` is reported as None (unverified), which the guard rejects.
    """
    def observed(name: str) -> bool:
        return int(window_row.get(f"{name}__missing", 1)) == 0

    def value(name: str) -> float:
        return float(window_row[name])

    stopped = False
    if observed("measured_linear") and observed("measured_angular") \
            and float(window_row["measured_linear__age_seconds"]) <= config.odom_max_age_seconds:
        stopped = (abs(value("measured_linear")) < config.stopped_linear_mps
                   and abs(value("measured_angular")) < config.stopped_angular_rads)
    localisation_poor = diagnosed == "localisation" or not observed("pose_covariance_trace") or (
        value("pose_covariance_trace") > config.localisation_poor_covariance_trace
    )
    planning_stale = diagnosed == "planning" or not observed("global_path_length") or (
        observed("replan_rate") and value("replan_rate") >= config.planning_stale_replan_rate_per_s
    )
    front_known = observed("minimum_front_range") and value("minimum_front_range") > 0.0
    immediate_collision_risk = bool(
        front_known and value("minimum_front_range") < config.immediate_collision_front_range_m
    )
    transient = bool(
        diagnosed == "frontal_blockage" and front_known
        and value("minimum_front_range") < config.transient_obstruction_front_range_m
    )

    def clearance(name: str) -> float | None:
        record = aux.get(name)
        if record is None:
            return None
        timestamp, measured = record
        if now - float(timestamp) > config.scan_max_age_seconds or not math.isfinite(measured):
            return None
        return float(measured)

    return RobotState(
        stopped=bool(stopped), stop_allowed=True, localisation_poor=bool(localisation_poor),
        planning_stale_or_blocked=bool(planning_stale),
        rear_clearance_m=clearance("rear_clearance_m"),
        rotation_clearance_m=clearance("rotation_clearance_m"),
        immediate_collision_risk=immediate_collision_risk,
        obstruction_may_be_transient=transient,
        relocalisation_available=bool(relocalisation_available),
        repeated_recovery_count=int(repeated_recovery_count),
    )


def scan_clearances(ranges: Sequence[float], angle_min: float, angle_increment: float) -> dict[str, float]:
    """Rear-sector (|angle| >= 5pi/6) and all-round minimum finite beam range."""
    rear = []
    every = []
    for index, raw in enumerate(ranges):
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            continue
        angle = angle_min + index * angle_increment
        angle = math.atan2(math.sin(angle), math.cos(angle))
        every.append(value)
        if abs(angle) >= 5 * math.pi / 6:
            rear.append(value)
    return {
        "rear_clearance_m": min(rear) if rear else float("nan"),
        "rotation_clearance_m": min(every) if every else float("nan"),
    }


@dataclass
class DecisionOutput:
    grid_index: int
    decision_index: int
    decision_time: float
    raw_score: float
    risk_score: float
    persistent: bool
    alarm: bool
    diagnosed_signal_group: str
    group_contributions: dict[str, float]
    window: np.ndarray
    prepared: np.ndarray
    compute_ms: float
    warning_id: str | None = None
    state: RobotState | None = None

    def record(self) -> dict[str, Any]:
        return {
            "decision_index": self.decision_index, "decision_time": self.decision_time,
            "raw_score": self.raw_score, "risk_score": self.risk_score,
            "persistent": self.persistent, "alarm": self.alarm,
            "diagnosed_signal_group": self.diagnosed_signal_group,
            "compute_ms": self.compute_ms, "warning_id": self.warning_id,
        }


class Scorer:
    """Frozen checkpoint + calibrator + alarm policy replay over the live stream."""

    def __init__(self, assets: MonitorAssets, feature_groups: Mapping[str, Sequence[str]],
                 *, threads: int = 1):
        from predict_decisions import LoadedModel  # torch is imported lazily inside
        import torch
        torch.set_num_threads(max(1, int(threads)))
        self.assets = assets
        self.model = LoadedModel(assets.model_dir, "cpu")
        if self.model.is_oracle:
            raise MonitorRefused("the analysis-only oracle can never run online")
        if self.model.checkpoint_sha256 != assets.checkpoint_sha256:
            raise MonitorRefused("loaded checkpoint differs from the resolved asset hash")
        self.calibrator = Calibrator.from_json_bytes(assets.calibrator_path.read_bytes())
        self.feature_names = tuple(self.model.feature_names)
        self.feature_groups = feature_groups
        self.policy = assets.alarm_policy
        self.rows: list[dict[str, float]] = []

    def score(self, window: np.ndarray, decision_time: float) -> tuple[float, float, bool, bool, np.ndarray]:
        prepared = self.model.prepare(window[None])
        raw = float(self.model.score_prepared(prepared, batch_size=1)[0])
        risk = float(self.calibrator.apply([raw])[0])
        self.rows.append({"decision_time": float(decision_time), "risk_score": risk})
        # Replaying the frozen policy over the whole stream keeps the exact offline
        # semantics (M-of-N persistence over consecutive decisions plus cooldown).
        last = apply_alarm_policy(self.rows, self.policy)[-1]
        return raw, risk, bool(last["persistent"]), bool(last["alarm"]), prepared[0]


class MonitorCore:
    """Everything the ROS node does except message I/O."""

    def __init__(
        self, *, run_id: str, policy_id: str, assets: MonitorAssets, root: Path,
        goal_x: float, goal_y: float, scorer: Scorer | None = None,
        state_config: StateEstimatorConfig = StateEstimatorConfig(),
        relocalisation_available: bool = False, clock: Callable[[], float] = time.perf_counter,
        feature_schema: Path | None = None, leakage_denylist: Path | None = None,
        failure_events: Path | None = None, ablations: Path | None = None,
    ):
        self.run_id = run_id
        self.policy_id = validate_policy_id(policy_id)
        self.forced_action = forced_action(policy_id)
        self.assets = assets
        self.root = root
        schema_path = feature_schema or root / "configs/feature_schema.yaml"
        contract = load_raw_feature_contract(schema_path)
        primary = load_primary_feature_set(schema_path)
        self.leakage_policy = LeakagePolicy.from_yaml(leakage_denylist or root / "configs/leakage_denylist.yaml")
        check_subscriptions(self.leakage_policy)
        windowing = yaml.safe_load(
            (failure_events or root / "configs/failure_events.yaml").read_text(encoding="utf-8")
        )["windowing"]
        self.windows = WindowBuilder(
            run_id=run_id, contract=contract, primary_features=primary,
            leakage_policy=self.leakage_policy, goal_x=goal_x, goal_y=goal_y,
            history_seconds=float(windowing["history_seconds"]),
            stride_seconds=float(windowing["decision_stride_seconds"]),
        )
        self.feature_groups = load_feature_groups(ablations or root / "configs/ablations.yaml")
        self.scorer = scorer
        self.state_config = state_config
        self.relocalisation_available = relocalisation_available
        self.clock = clock
        self.aux: dict[str, tuple[float, float]] = {}
        self.decisions: list[dict[str, Any]] = []
        self.alarms: list[dict[str, Any]] = []
        self.compute_ms: list[float] = []
        self.mission_start: float | None = None

    # -- inputs
    def receive(self, topic: str, message: Any, timestamp: float) -> int:
        return self.windows.buffer.add_message(topic, message, timestamp)

    def receive_scan_clearances(self, ranges: Sequence[float], angle_min: float,
                                angle_increment: float, timestamp: float) -> None:
        for name, value in scan_clearances(ranges, angle_min, angle_increment).items():
            self.aux[name] = (float(timestamp), float(value))

    def mission_started(self, timestamp: float) -> bool:
        if self.mission_start is not None:
            return False
        self.mission_start = float(timestamp)
        self.windows.start(self.mission_start)
        return True

    # -- decisions
    def step(self, now: float) -> list[DecisionOutput]:
        outputs = []
        for grid_index, moment in self.windows.due_grid_times(now):
            started = self.clock()
            window = self.windows.advance(grid_index, moment)
            if window is None:
                continue
            if self.scorer is None:
                raise MonitorRefused("no scorer attached")
            raw, risk, persistent, alarm, prepared = self.scorer.score(window, moment)
            diagnosed, contributions = diagnose_signal_group(
                prepared[-1], self.scorer.feature_names, self.feature_groups,
            )
            elapsed_ms = (self.clock() - started) * 1e3
            output = DecisionOutput(
                grid_index=grid_index, decision_index=self.windows.decision_index(grid_index),
                decision_time=moment, raw_score=raw, risk_score=risk, persistent=persistent,
                alarm=alarm, diagnosed_signal_group=diagnosed, group_contributions=contributions,
                window=window, prepared=prepared, compute_ms=elapsed_ms,
            )
            if alarm:
                output.warning_id = f"{self.run_id}-w{len(self.alarms) + 1:03d}"
                output.state = estimate_robot_state(
                    self.windows.enriched_tail[-1], diagnosed=diagnosed, aux=self.aux, now=moment,
                    config=self.state_config,
                    relocalisation_available=self.relocalisation_available,
                    repeated_recovery_count=len(self.alarms),
                )
                self.alarms.append({
                    **output.record(), "threshold": self.assets.alarm_policy.threshold,
                    "state": asdict(output.state), "group_contributions": contributions,
                })
            self.compute_ms.append(elapsed_ms)
            self.decisions.append(output.record())
            outputs.append(output)
        return outputs

    def request_values(self, output: DecisionOutput) -> dict[str, str]:
        """Key/value payload of one ``/research2/recovery_requests`` message."""
        assert output.state is not None and output.warning_id is not None
        state = asdict(output.state)
        values = {
            "run_id": self.run_id, "warning_id": output.warning_id,
            "policy_id": self.policy_id,
            "risk_score": repr(output.risk_score), "raw_score": repr(output.raw_score),
            "threshold": repr(self.assets.alarm_policy.threshold),
            "decision_index": str(output.decision_index),
            "decision_time": repr(output.decision_time),
            "diagnosed_signal_group": output.diagnosed_signal_group,
            "engineering_smoke": "true" if self.assets.engineering_smoke else "false",
        }
        for key, value in state.items():
            if isinstance(value, bool):
                values[key] = "true" if value else "false"
            elif value is None:
                values[key] = "none"
            else:
                values[key] = repr(value) if isinstance(value, float) else str(value)
        if self.forced_action:
            values["forced_action"] = self.forced_action
        return values

    def sidecar(self) -> dict[str, Any]:
        latencies = np.asarray(self.compute_ms, dtype=float)
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "policy_id": self.policy_id,
            "forced_action": self.forced_action,
            "engineering_smoke": self.assets.engineering_smoke,
            "assets": self.assets.record(),
            "subscribed_topics": list(SUBSCRIBED_TOPICS),
            "mission_start": self.mission_start,
            "grid_start": self.windows.grid_start,
            "history_seconds": self.windows.history_seconds,
            "stride_seconds": self.windows.stride,
            "decision_count": len(self.decisions),
            "alarm_count": len(self.alarms),
            "alarms": self.alarms,
            "decisions": self.decisions,
            "state_estimator": asdict(self.state_config),
            "relocalisation_available": self.relocalisation_available,
            "buffer": self.windows.buffer.counters(),
            "compute_ms": {
                "median": float(np.median(latencies)) if latencies.size else None,
                "p95": float(np.percentile(latencies, 95)) if latencies.size else None,
                "max": float(latencies.max()) if latencies.size else None,
            },
            "causal_role": "label_only",
        }


def write_sidecar(document: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "DecisionOutput", "EVENT_TOPIC", "FEATURE_TOPICS", "GROUP_TO_SIGNAL", "MISSION_START_TOPIC",
    "MonitorAssets", "MonitorCore", "MonitorRefused", "REQUEST_TOPIC", "SUBSCRIBED_TOPICS",
    "SampleBuffer", "Scorer", "StateEstimatorConfig", "WARNING_TOPIC", "WindowBuilder",
    "check_subscriptions", "diagnose_signal_group", "estimate_robot_state", "grid_time",
    "load_alarm_policy", "load_feature_groups", "resolve_monitor_assets", "scan_clearances",
    "write_sidecar",
]

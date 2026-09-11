"""Causal adapter for split-safe Research 1 development bags.

Research 1 bags carry no ``/research2/events`` stream and no Research 2 fault
injection. This module decides, mechanically and per episode, whether a retained
Research 1 development bag can be expressed as a Research 2-equivalent episode:

* every raw-feature topic that the frozen feature schema needs is present;
* an unambiguous episode start and end exist on one time base;
* the Research 1 aggregate outcome maps onto a Research 2 event class with an
  explicit, exact-or-bounded event time;
* the aggregate row and the bag agree with each other.

Time base. Research 1 recorded its bags with wall-clock receive timestamps while
every header stamp is simulation time, and the campaign real-time factor varied
between episodes. Research 2 defines windows, horizons and guards in simulation
seconds, so the adapter builds a per-episode monotone receive-to-simulation clock map
from ``/odom`` (header stamp against recorder receive time) and expresses every
availability time in simulation seconds. Header-stamped label evidence is used as is.

Nothing here reads a Research 1 ``test_*`` map, a confirmatory row, or a validation
bag payload. Validation rows are counted only. Nothing is written into Research 1.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import csv
from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.labels.operational_events import (
    MotionSample,
    PoseErrorSample,
    first_immobilisation,
    first_localisation_loss,
    first_primary_event,
)


ADAPTER_VERSION = 1
CAMPAIGN_ID = "research1_development_v1"
RESEARCH1_ROOT = Path("/home/eao/risk-calibrated-nav")
RECORDING_PROFILE = "research1_g6_wall_clock_receive"
TIME_BASE = "simulation_time_via_receive_clock_map"
EVENT_SOURCE = "adapter_sidecar"

REQUIRED_FEATURE_TOPICS = ("/cmd_vel", "/odom", "/amcl_pose", "/scan", "/plan")
OPTIONAL_FEATURE_TOPICS = (
    "/local_plan", "/semantic/confidence", "/semantic/uncertainty",
    "/semantic/inference_latency_ms", "/research2/features/perception",
)
LABEL_EVIDENCE_TOPICS = ("/ground_truth_pose", "/behavior_tree_log")
STREAMED_TOPICS = (
    "/cmd_vel", "/odom", "/amcl_pose", "/ground_truth_pose", "/plan",
    "/behavior_tree_log", "/collision_event",
)

RESEARCH2_FAULT_FAMILIES = frozenset({
    "camera_occlusion", "lidar_dropout", "wheel_slip", "localisation_perturbation",
    "dynamic_blockage", "planner_oscillation", "semantic_corruption",
})

# Research 1 terminal_state -> Research 2 terminal event class (None = no event).
TERMINAL_EVENT_CLASS: dict[str, str | None] = {
    "collision": "collision",
    "planner_failure": "navigation_abort",
    "timeout": "mission_timeout",
    "false_arrival": "false_arrival",
    "success": None,
}

EPISODE_START_DEFINITION = (
    "first /behavior_tree_log message (Nav2 bt_navigator ticks only after the "
    "NavigateToPose goal is accepted); receive time mapped to simulation seconds. "
    "Research 1 stamps sim_t0 immediately before send_goal, so the true dispatch "
    "precedes this by the acceptance latency (start_lag recorded per episode)."
)

EVENT_TIME_DERIVATION: dict[str, dict[str, str]] = {
    "collision": {
        "method": "exact_first_collision_event_message",
        "detail": (
            "first /collision_event message carrying at least one contact; Research 1 "
            "republishes only disqualifying full-body contacts on this topic. Cross-checked "
            "against episode_start + aggregate duration_s (residual recorded)."
        ),
        "exactness": "exact",
    },
    "navigation_abort": {
        "method": "bounded_episode_start_plus_aggregate_duration",
        "detail": (
            "Research 1 does not log the root NavigateToPose ABORTED transition; the "
            "event time is episode_start + duration_s where duration_s is the Research 1 "
            "simulated-seconds duration from goal dispatch to the aborted action result. "
            "Upper bound: overestimates by the dispatch-to-first-tick lag plus the 0.1 s "
            "controller poll interval."
        ),
        "exactness": "bounded_upper",
    },
    "mission_timeout": {
        "method": "route_budget_boundary_via_aggregate_duration",
        "detail": (
            "episode_start + duration_s, required to lie within tolerance of the frozen "
            "route episode_timeout_s (simulated seconds). Wall-clock backstop timeouts "
            "that fire before the simulated budget are rejected."
        ),
        "exactness": "bounded_upper",
    },
    "false_arrival": {
        "method": "bounded_episode_start_plus_aggregate_duration",
        "detail": (
            "episode_start + duration_s at the Nav2 SUCCEEDED result, with label-only "
            "ground-truth goal distance required to exceed the frozen goal tolerance."
        ),
        "exactness": "bounded_upper",
    },
    "none": {
        "method": "episode_end_at_episode_start_plus_aggregate_duration",
        "detail": "success has no event; the episode ends at the SUCCEEDED result.",
        "exactness": "bounded_upper",
    },
}

# Research 1 commit 07c254b (2026-08-26, protocol v1.21) redefined duration_s from wall
# seconds to simulated seconds. Descendant commits report simulated durations; earlier
# commits report wall durations that the clock map converts to simulation seconds.
DURATION_SIM_TIME_COMMIT = "07c254b0b583b77760cfaee06e69dcbb6a718130"
DURATION_TIME_BASES = ("simulation", "wall_clock")


@dataclass(frozen=True)
class AdapterThresholds:
    history_seconds: float = 5.0
    duration_tolerance_seconds: float = 2.0
    collision_residual_tolerance_seconds: float = 1.0
    timeout_budget_tolerance_seconds: float = 1.0
    telemetry_tail_tolerance_seconds: float = 0.5
    minimum_clock_knots: int = 10
    pose_match_gap_seconds: float = 0.5
    command_freshness_seconds: float = 0.5


@dataclass(frozen=True)
class RouteSpec:
    map_id: str
    route_id: str
    start: dict[str, float]
    goal: dict[str, float]
    shortest_path_m: float
    goal_tolerance_m: float
    episode_timeout_s: float


@dataclass
class BagEvidence:
    """Everything the adapter needs from one bag, on the recorder's own clocks."""

    topic_counts: dict[str, int]
    first_receive: dict[str, float]
    last_receive: dict[str, float]
    clock_knots: list[tuple[float, float]]
    first_contact_receive: float | None = None
    last_nonzero_command_receive: float | None = None
    commands: list[tuple[float, float]] = field(default_factory=list)
    odometry: list[tuple[float, float, float]] = field(default_factory=list)
    ground_truth: list[tuple[float, float, float, float]] = field(default_factory=list)
    amcl: list[tuple[float, float, float, float]] = field(default_factory=list)
    mcap_files: int = 1


class ReceiveClockMap:
    """Monotone piecewise-linear map from recorder receive time to simulation time."""

    def __init__(self, knots: Sequence[tuple[float, float]]):
        cleaned: list[tuple[float, float]] = []
        for wall, sim in knots:
            wall, sim = float(wall), float(sim)
            if not (math.isfinite(wall) and math.isfinite(sim)):
                continue
            if cleaned and (wall <= cleaned[-1][0] or sim < cleaned[-1][1]):
                continue
            cleaned.append((wall, sim))
        if len(cleaned) < 2:
            raise ValueError("a receive clock map needs at least two monotone knots")
        self._wall = [item[0] for item in cleaned]
        self._sim = [item[1] for item in cleaned]

    @property
    def knots(self) -> list[tuple[float, float]]:
        return list(zip(self._wall, self._sim))

    @property
    def real_time_factor(self) -> float:
        """Least-squares simulated seconds per wall second over the whole map."""
        n = len(self._wall)
        mean_w = sum(self._wall) / n
        mean_s = sum(self._sim) / n
        denominator = sum((w - mean_w) ** 2 for w in self._wall)
        if denominator == 0:
            return float("nan")
        return sum((w - mean_w) * (s - mean_s) for w, s in zip(self._wall, self._sim)) / denominator

    def to_sim(self, wall_seconds: float) -> float:
        wall = float(wall_seconds)
        index = bisect_left(self._wall, wall)
        if index <= 0:
            left, right = 0, 1
        elif index >= len(self._wall):
            left, right = len(self._wall) - 2, len(self._wall) - 1
        else:
            left, right = index - 1, index
        w0, w1 = self._wall[left], self._wall[right]
        s0, s1 = self._sim[left], self._sim[right]
        return s0 + (wall - w0) * (s1 - s0) / (w1 - w0)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "time_base": TIME_BASE,
            "source_topic": "/odom",
            "knot_fields": ["receive_wall_seconds", "simulation_seconds"],
            "real_time_factor": self.real_time_factor,
            "knots": [[w, s] for w, s in zip(self._wall, self._sim)],
        }

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> "ReceiveClockMap":
        return cls([(item[0], item[1]) for item in document["knots"]])


def research2_fault_family(shift_family: str) -> str:
    """Label-only family for Research 2 purposes; never one of the seven R2 families."""
    shift = str(shift_family).strip()
    if not shift:
        raise ValueError("Research 1 shift_family is empty")
    family = "none" if shift == "clean" else f"research1_{shift}"
    if family in RESEARCH2_FAULT_FAMILIES:
        raise ValueError(f"Research 1 family collides with a Research 2 family: {family}")
    return family


def research2_severity(shift_family: str, severity: str | int) -> str:
    return "none" if str(shift_family) == "clean" else f"research1_{int(severity)}"


def episode_key(run_id: str) -> str:
    return f"r1-{run_id}"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_aggregates(research1_root: Path) -> dict[str, Path]:
    """Map run_id -> aggregate CSV path without opening any aggregate."""
    return {path.stem: path for path in sorted((research1_root / "results/raw").rglob("*.csv"))}


def load_aggregate(path: Path, expected_run_id: str, expected_map_id: str) -> dict[str, str]:
    """Read one development aggregate row. Refuses protected maps before reading."""
    if expected_map_id.startswith("test_") or not expected_map_id.startswith("dev_"):
        raise ValueError(f"refusing to read a non-development Research 1 aggregate: {expected_map_id}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"aggregate must contain exactly one row: {path}")
    row = rows[0]
    if row.get("run_id") != expected_run_id:
        raise ValueError(f"aggregate run_id differs from catalog: {path}")
    if row.get("map_id") != expected_map_id or str(row.get("map_id", "")).startswith("test_"):
        raise ValueError(f"aggregate map_id differs from catalog or is protected: {path}")
    return row


def load_route_spec(research1_root: Path, map_id: str, route_id: str) -> RouteSpec | None:
    if not map_id.startswith("dev_"):
        raise ValueError(f"refusing non-development route specification: {map_id}")
    path = research1_root / "configs/routes" / f"{map_id}.yaml"
    if not path.is_file():
        return None
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for route in document.get("routes", []):
        if route.get("route_id") == route_id:
            return RouteSpec(
                map_id=map_id,
                route_id=route_id,
                start={key: float(value) for key, value in route["start"].items()},
                goal={key: float(value) for key, value in route["goal"].items()},
                shortest_path_m=float(route.get("shortest_path_m") or 0.0),
                goal_tolerance_m=float(document.get("goal_tolerance_m", 0.25)),
                episode_timeout_s=float(document.get("episode_timeout_s") or 180.0),
            )
    return None


def yaw_from_quaternion(orientation) -> float:
    return math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )


def _yaw_error(left: float, right: float) -> float:
    return abs(math.atan2(math.sin(left - right), math.cos(left - right)))


def _nearest_pose(
    timestamp: float, timestamps: list[float], poses: list[tuple[float, float, float]],
    maximum_gap_seconds: float,
) -> tuple[float, float, float] | None:
    index = bisect_left(timestamps, timestamp)
    candidates = [c for c in (index - 1, index, index + 1) if 0 <= c < len(timestamps)]
    if not candidates:
        return None
    selected = min(candidates, key=lambda c: abs(timestamps[c] - timestamp))
    if abs(timestamps[selected] - timestamp) > maximum_gap_seconds:
        return None
    return poses[selected]


def build_pose_error_samples(
    ground_truth: Sequence[tuple[float, float, float, float]],
    amcl: Sequence[tuple[float, float, float, float]],
    *, maximum_gap_seconds: float = 0.5,
) -> list[PoseErrorSample]:
    """Mirror scripts/derive_operational_events.read_evidence pose matching."""
    truth = sorted(ground_truth)
    times = [item[0] for item in truth]
    poses = [(item[1], item[2], item[3]) for item in truth]
    samples: list[PoseErrorSample] = []
    previous = float("-inf")
    for timestamp, x, y, estimate_yaw in sorted(amcl):
        match = _nearest_pose(timestamp, times, poses, maximum_gap_seconds)
        if match is None or timestamp <= previous:
            continue
        previous = timestamp
        samples.append(PoseErrorSample(
            timestamp=timestamp,
            translation_error_m=math.dist((x, y), match[:2]),
            yaw_error_rad=_yaw_error(estimate_yaw, match[2]),
        ))
    return samples


def build_motion_samples(
    commands: Sequence[tuple[float, float]],
    odometry: Sequence[tuple[float, float, float]],
    *, command_freshness_seconds: float = 0.5,
) -> list[MotionSample]:
    """Mirror scripts/derive_operational_events.read_evidence command matching."""
    ordered_commands = sorted(commands)
    command_times = [item[0] for item in ordered_commands]
    samples: list[MotionSample] = []
    previous = float("-inf")
    for timestamp, x, y in sorted(odometry):
        index = bisect_right(command_times, timestamp) - 1
        if index < 0 or timestamp <= previous:
            continue
        command_time, linear = ordered_commands[index]
        if timestamp - command_time > command_freshness_seconds:
            continue
        previous = timestamp
        samples.append(MotionSample(timestamp, linear, x, y))
    return samples


def read_bag_evidence(bag_dir: Path) -> BagEvidence:
    """Stream only the adapter topics from one MCAP bag (rosbag2_py required)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    metadata = reader.get_metadata()
    topic_counts = {
        str(item.topic_metadata.name): int(item.message_count)
        for item in metadata.topics_with_message_count
    }
    reader.set_filter(rosbag2_py.StorageFilter(
        topics=[topic for topic in STREAMED_TOPICS if topic in types]
    ))
    evidence = BagEvidence(
        topic_counts=topic_counts, first_receive={}, last_receive={}, clock_knots=[],
        mcap_files=len(metadata.relative_file_paths),
    )
    while reader.has_next():
        topic, data, receive_ns = reader.read_next()
        receive = receive_ns * 1e-9
        evidence.first_receive.setdefault(topic, receive)
        evidence.last_receive[topic] = receive
        if topic == "/odom":
            message = deserialize_message(data, get_message(types[topic]))
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            evidence.clock_knots.append((receive, stamp))
            evidence.odometry.append((
                stamp, float(message.pose.pose.position.x), float(message.pose.pose.position.y),
            ))
        elif topic == "/cmd_vel":
            message = deserialize_message(data, get_message(types[topic]))
            linear = float(message.linear.x)
            evidence.commands.append((receive, linear))
            if abs(linear) > 1e-6 or abs(float(message.angular.z)) > 1e-6:
                evidence.last_nonzero_command_receive = receive
        elif topic == "/ground_truth_pose":
            message = deserialize_message(data, get_message(types[topic]))
            evidence.ground_truth.append((
                message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
                float(message.pose.position.x), float(message.pose.position.y),
                yaw_from_quaternion(message.pose.orientation),
            ))
        elif topic == "/amcl_pose":
            message = deserialize_message(data, get_message(types[topic]))
            evidence.amcl.append((
                message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
                float(message.pose.pose.position.x), float(message.pose.pose.position.y),
                yaw_from_quaternion(message.pose.pose.orientation),
            ))
        elif topic == "/collision_event" and evidence.first_contact_receive is None:
            message = deserialize_message(data, get_message(types[topic]))
            if len(message.contacts) > 0:
                evidence.first_contact_receive = receive
    return evidence


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def assess_episode(
    catalog_row: Mapping[str, Any],
    aggregate: Mapping[str, str],
    route: RouteSpec | None,
    evidence: BagEvidence,
    *,
    event_config: Mapping[str, Any],
    thresholds: AdapterThresholds = AdapterThresholds(),
    duplicate_run_id: bool = False,
    duration_time_base: str | None = None,
    research1_campaign: str | None = None,
) -> dict[str, Any]:
    """Decide admissibility for one development row; pure given the evidence.

    ``duration_time_base`` declares whether the aggregate ``duration_s`` is in
    simulated seconds or wall seconds (from Research 1 commit ancestry). When it is
    None both bases are tried against the bag and the row is rejected unless exactly
    one is consistent.
    """
    reasons: list[str] = []
    run_id = str(catalog_row["run_id"])
    counts = evidence.topic_counts
    present = {topic for topic, count in counts.items() if count > 0}
    missing_channels = [t for t in OPTIONAL_FEATURE_TOPICS if t not in present]
    for topic in REQUIRED_FEATURE_TOPICS:
        if topic not in present:
            reasons.append(f"missing_required_topic:{topic}")
    for topic in LABEL_EVIDENCE_TOPICS:
        if topic not in present:
            reasons.append(f"missing_label_evidence_topic:{topic}")
    if evidence.mcap_files != 1:
        reasons.append("bag_is_not_a_single_mcap_file")
    if duplicate_run_id:
        reasons.append("duplicate_run_id")
    for field_name in ("map_id", "route_id", "system_id", "seed"):
        if str(aggregate.get(field_name, "")) != str(catalog_row.get(field_name, "")):
            reasons.append(f"identity_mismatch:{field_name}")
    terminal_state = str(aggregate.get("terminal_state", ""))
    if str(aggregate.get("invalid_reason", "")).strip() or terminal_state == "invalid":
        reasons.append("aggregate_invalid")
    if terminal_state not in TERMINAL_EVENT_CLASS and terminal_state != "invalid":
        reasons.append(f"unknown_terminal_state:{terminal_state}")
    if route is None:
        reasons.append("route_spec_missing")
    shift_family = str(aggregate.get("shift_family", ""))
    try:
        fault_family = research2_fault_family(shift_family)
    except ValueError:
        fault_family = None
        reasons.append("shift_family_unmappable")

    clock: ReceiveClockMap | None = None
    if len(evidence.clock_knots) < thresholds.minimum_clock_knots:
        reasons.append("clock_map_insufficient_odometry")
    else:
        sims = [item[1] for item in evidence.clock_knots]
        if any(later < earlier for earlier, later in zip(sims, sims[1:])):
            reasons.append("simulation_clock_not_monotone")
        else:
            clock = ReceiveClockMap(evidence.clock_knots)

    duration_s = _float_or_none(aggregate.get("duration_s"))
    goal_distance = _float_or_none(aggregate.get("goal_distance_gt_m"))
    result: dict[str, Any] = {
        "run_id": run_id,
        "shift_family": shift_family,
        "research1_severity": str(aggregate.get("severity", "")),
        "fault_family": fault_family,
        "severity": research2_severity(shift_family, aggregate.get("severity", 0))
        if fault_family else None,
        "terminal_state": terminal_state,
        "success": str(aggregate.get("success", "")).lower() == "true",
        "collision": str(aggregate.get("collision", "")).lower() == "true",
        "timeout": str(aggregate.get("timeout", "")).lower() == "true",
        "duration_s": duration_s,
        "goal_distance_gt_m": goal_distance,
        "mapped_event_class": TERMINAL_EVENT_CLASS.get(terminal_state),
        "missing_channels": missing_channels,
        "topic_message_counts": {t: counts.get(t, 0) for t in REQUIRED_FEATURE_TOPICS
                                 + OPTIONAL_FEATURE_TOPICS + LABEL_EVIDENCE_TOPICS
                                 + ("/collision_event",)},
        "real_time_factor": clock.real_time_factor if clock else None,
        "clock_knots": len(evidence.clock_knots),
        "research1_campaign": research1_campaign,
        "research1_commit_sha": aggregate.get("commit_sha"),
        "aggregate_duration_time_base": duration_time_base,
        "duration_time_base_source": "commit_ancestry" if duration_time_base else "bag_evidence",
    }
    if duration_time_base is not None and duration_time_base not in DURATION_TIME_BASES:
        reasons.append(f"unknown_duration_time_base:{duration_time_base}")
    if clock is None or duration_s is None or route is None or not present >= set(REQUIRED_FEATURE_TOPICS):
        result.update({"admissible": False, "reasons": reasons})
        return result

    bag_start_sim = clock.to_sim(min(evidence.first_receive.values()))
    bag_end_sim = clock.to_sim(max(evidence.last_receive.values()))
    start_method = None
    start_sim = None
    if "/behavior_tree_log" in evidence.first_receive:
        start_sim = clock.to_sim(evidence.first_receive["/behavior_tree_log"])
        start_method = "first_behavior_tree_tick_after_goal_acceptance"
    elif "/plan" in evidence.first_receive:
        start_sim = clock.to_sim(evidence.first_receive["/plan"])
        start_method = "first_global_plan_fallback"
    else:
        reasons.append("no_episode_start_evidence")
    first_plan_sim = (
        clock.to_sim(evidence.first_receive["/plan"]) if "/plan" in evidence.first_receive else None
    )
    result.update({
        "bag_start_sim": bag_start_sim, "bag_end_sim": bag_end_sim,
        "episode_start_sim": start_sim, "episode_start_method": start_method,
        "start_lag_after_bag_start_s": None if start_sim is None else start_sim - bag_start_sim,
        "first_plan_minus_start_s": (
            None if start_sim is None or first_plan_sim is None else first_plan_sim - start_sim
        ),
    })
    if start_sim is None:
        result.update({"admissible": False, "reasons": reasons})
        return result

    start_wall = (
        evidence.first_receive["/behavior_tree_log"]
        if "/behavior_tree_log" in evidence.first_receive else evidence.first_receive["/plan"]
    )
    last_command_sim = (
        clock.to_sim(evidence.last_receive["/cmd_vel"]) if "/cmd_vel" in evidence.last_receive else None
    )
    last_odom_sim = evidence.odometry[-1][0] if evidence.odometry else None
    terminal_by_base = {
        "simulation": start_sim + duration_s,
        "wall_clock": clock.to_sim(start_wall + duration_s),
    }

    def _consistent(candidate: float) -> bool:
        if candidate > bag_end_sim + thresholds.telemetry_tail_tolerance_seconds:
            return False
        if last_command_sim is not None and abs(candidate - last_command_sim) > thresholds.duration_tolerance_seconds:
            return False
        return True

    if duration_time_base in DURATION_TIME_BASES:
        chosen_base = duration_time_base
    else:
        consistent = [base for base in DURATION_TIME_BASES if _consistent(terminal_by_base[base])]
        if len(consistent) == 1:
            chosen_base = consistent[0]
        else:
            chosen_base = "simulation"
            reasons.append("aggregate_duration_time_base_ambiguous")
    result["aggregate_duration_time_base"] = chosen_base
    result["terminal_by_duration_base_sim"] = terminal_by_base
    terminal_sim = terminal_by_base[chosen_base]
    duration_residual = None if last_command_sim is None else terminal_sim - last_command_sim
    if terminal_sim > bag_end_sim + thresholds.telemetry_tail_tolerance_seconds:
        reasons.append("terminal_beyond_bag_end")
    if duration_residual is not None and abs(duration_residual) > thresholds.duration_tolerance_seconds:
        reasons.append("aggregate_duration_inconsistent_with_bag")
    if last_odom_sim is not None and last_odom_sim < terminal_sim - thresholds.telemetry_tail_tolerance_seconds:
        reasons.append("telemetry_ends_before_terminal")

    event_class = TERMINAL_EVENT_CLASS.get(terminal_state)
    event_time: float | None = None
    method = EVENT_TIME_DERIVATION[event_class or "none"]
    collision_residual = None
    episode_end = terminal_sim
    if terminal_state == "collision":
        if evidence.first_contact_receive is None:
            reasons.append("collision_without_contact_evidence")
        else:
            event_time = clock.to_sim(evidence.first_contact_receive)
            collision_residual = event_time - terminal_sim
            if abs(collision_residual) > thresholds.collision_residual_tolerance_seconds:
                reasons.append("collision_time_inconsistent_with_aggregate")
            if event_time < start_sim:
                reasons.append("collision_before_episode_start")
            episode_end = max(terminal_sim, event_time)
    elif terminal_state == "planner_failure":
        event_time = terminal_sim
    elif terminal_state == "timeout":
        event_time = terminal_sim
        if duration_s < route.episode_timeout_s - thresholds.timeout_budget_tolerance_seconds:
            reasons.append("timeout_not_at_route_budget")
    elif terminal_state == "false_arrival":
        event_time = terminal_sim
        if goal_distance is None or goal_distance <= route.goal_tolerance_m:
            reasons.append("false_arrival_within_goal_tolerance")
    elif terminal_state == "success":
        if goal_distance is None or goal_distance > route.goal_tolerance_m:
            reasons.append("success_outside_goal_tolerance")
    if episode_end - start_sim < thresholds.history_seconds:
        reasons.append("episode_shorter_than_history_window")

    localisation = event_config["events"]["localisation_loss"]
    immobilisation = event_config["events"]["immobilisation"]
    pose_errors = build_pose_error_samples(
        evidence.ground_truth, evidence.amcl,
        maximum_gap_seconds=thresholds.pose_match_gap_seconds,
    )
    motion = build_motion_samples(
        [(clock.to_sim(receive), linear) for receive, linear in evidence.commands],
        evidence.odometry, command_freshness_seconds=thresholds.command_freshness_seconds,
    )
    candidates: dict[str, float | None] = {
        "localisation_loss": first_localisation_loss(
            pose_errors,
            translation_threshold_m=float(localisation["translation_error_m"]),
            yaw_threshold_rad=float(localisation["yaw_error_rad"]),
            persistence_seconds=float(localisation["persistence_seconds"]),
            maximum_gap_seconds=float(localisation["maximum_evidence_gap_seconds"]),
        ),
        "immobilisation": first_immobilisation(
            motion,
            command_threshold_mps=float(immobilisation["minimum_command_speed_mps"]),
            maximum_progress_m=float(immobilisation["maximum_progress_m"]),
            persistence_seconds=float(immobilisation["persistence_seconds"]),
            maximum_gap_seconds=float(immobilisation["maximum_evidence_gap_seconds"]),
        ),
    }
    if event_class and event_time is not None:
        candidates[event_class] = event_time
    primary = first_primary_event(candidates, event_config["event_precedence"])
    if primary is not None and primary[1] < start_sim:
        reasons.append("operational_event_before_episode_start")
    if primary is not None and primary[1] > episode_end:
        reasons.append("operational_event_after_episode_end")

    result.update({
        "episode_end_sim": episode_end,
        "terminal_event_class": event_class,
        "terminal_event_time_sim": event_time,
        "terminal_time_method": method["method"],
        "terminal_time_exactness": method["exactness"],
        "collision_residual_s": collision_residual,
        "duration_residual_s": duration_residual,
        "route_episode_timeout_s": route.episode_timeout_s,
        "route_goal_tolerance_m": route.goal_tolerance_m,
        "candidate_event_times": candidates,
        "primary_event_class": primary[0] if primary else None,
        "primary_event_time_sim": primary[1] if primary else None,
        "pose_error_samples": len(pose_errors),
        "motion_samples": len(motion),
        "admissible": not reasons,
        "reasons": reasons,
    })
    return result


# --------------------------------------------------------------------------- adaptation


def _stamp(seconds: float) -> dict[str, int]:
    whole = math.floor(seconds)
    return {"sec": int(whole), "nanosec": int(round((seconds - whole) * 1e9))}


def build_event_sidecar(assessment: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Research 2-style event list for an adapted episode (label-only)."""
    start = float(assessment["episode_start_sim"])
    end = float(assessment["episode_end_sim"])
    events = [
        {"event_type": "recording_window_started", "parameters": {}, "reason": "",
         "stamp": _stamp(float(assessment["bag_start_sim"]))},
        {"event_type": "episode_started", "parameters": {}, "reason": "", "stamp": _stamp(start)},
        {"event_type": "no_injection_control" if assessment["fault_family"] == "none"
         else "research1_shift_condition",
         "parameters": {} if assessment["fault_family"] == "none" else {
             "shift_family": assessment["shift_family"],
             "severity": assessment["research1_severity"],
             "present_from_episode_start": True,
         },
         "reason": "", "stamp": _stamp(start)},
        {"event_type": "goal_dispatched",
         "parameters": {"derivation": assessment["episode_start_method"]},
         "reason": "", "stamp": _stamp(start)},
        {"event_type": "terminal_event",
         "parameters": {
             "terminal_state": assessment["terminal_state"],
             "success": bool(assessment["success"]),
             "goal_distance_gt_m": assessment["goal_distance_gt_m"],
             "terminal_event_class": assessment["terminal_event_class"],
             "terminal_event_time": assessment["terminal_event_time_sim"],
             "terminal_time_method": assessment["terminal_time_method"],
             "terminal_time_exactness": assessment["terminal_time_exactness"],
         },
         "reason": assessment["terminal_state"],
         "stamp": _stamp(
             float(assessment["terminal_event_time_sim"])
             if assessment["terminal_event_time_sim"] is not None else end
         )},
        {"event_type": "recording_window_ended", "parameters": {}, "reason": "",
         "stamp": _stamp(float(assessment["bag_end_sim"]))},
    ]
    return events


def build_summary(
    *,
    catalog_row: Mapping[str, Any],
    aggregate: Mapping[str, str],
    route: RouteSpec,
    assessment: Mapping[str, Any],
    bag_dir: Path,
    bag_checksum_sha256: str,
    aggregate_path: Path,
    aggregate_sha256: str,
    event_sidecar: Path,
    topic_health_sidecar: Path,
    clock_map_path: Path,
    clock_map_sha256: str,
    adapter_config_hash: str,
) -> dict[str, Any]:
    run_id = str(catalog_row["run_id"])
    goal_yaw = math.atan2(route.goal["y"] - route.start["y"], route.goal["x"] - route.start["x"])
    try:
        shift_parameters = json.loads(aggregate.get("shift_params_json") or "{}")
    except json.JSONDecodeError:
        shift_parameters = {"unparsed": aggregate.get("shift_params_json")}
    return {
        "schema_version": 1,
        "identity": {
            "run_id": run_id,
            "campaign_id": CAMPAIGN_ID,
            "episode_key": episode_key(run_id),
            "timestamp_utc": aggregate.get("timestamp"),
            "protocol_version": "1.0",
            "replacement_for_episode_key": None,
            "replacement_for_run_id": None,
            "parent_research1_run_id": run_id,
            "research1_platform_commit": aggregate.get("commit_sha"),
            "research1_repository_head": aggregate.get("commit_sha"),
            "research2_config_hash": adapter_config_hash,
            "adapter_version": ADAPTER_VERSION,
        },
        "environment": {
            "map_id": route.map_id,
            "route_id": route.route_id,
            "start_pose": {"x": route.start["x"], "y": route.start["y"],
                           "yaw": route.start.get("yaw", 0.0)},
            "goal_pose": {"x": route.goal["x"], "y": route.goal["y"], "yaw": goal_yaw},
            "seed": int(aggregate.get("seed") or 0),
            "system_id": aggregate.get("system_id"),
            "split": "development",
            "protected_test_used": False,
        },
        "label_only": {
            "fault_family": assessment["fault_family"],
            "severity": assessment["severity"],
            "parameters": shift_parameters,
            "planned_onset_seconds": None,
            "clean_prefix_seconds": None,
            "primary_event_class": assessment["primary_event_class"],
            "perception_metrics": None,
            "placement": None,
            "research1_adapted": True,
            "research1_shift": {
                "shift_family": assessment["shift_family"],
                "severity": assessment["research1_severity"],
                "present_from_episode_start": True,
                "is_research2_injection": False,
            },
            "research1_campaign": assessment.get("research1_campaign"),
            "aggregate_duration_time_base": assessment.get("aggregate_duration_time_base"),
            "terminal_event_class": assessment["terminal_event_class"],
            "terminal_event_time": assessment["terminal_event_time_sim"],
            "terminal_time_method": assessment["terminal_time_method"],
            "terminal_time_exactness": assessment["terminal_time_exactness"],
            "episode_start_method": assessment["episode_start_method"],
            "candidate_event_times": assessment["candidate_event_times"],
            "missing_channels": assessment["missing_channels"],
            "real_time_factor": assessment["real_time_factor"],
        },
        "outcome": {
            "terminal_state": assessment["terminal_state"],
            "success": bool(assessment["success"]),
            "collision": bool(assessment["collision"]),
            "timeout": bool(assessment["timeout"]),
            "invalid_reason": None,
            "duration_s": assessment["duration_s"],
            "path_length_m": _float_or_none(aggregate.get("path_length_m")),
            "shortest_path_m": _float_or_none(aggregate.get("shortest_path_m")),
            "spl": _float_or_none(aggregate.get("spl")),
            "minimum_clearance_m": _float_or_none(aggregate.get("minimum_clearance_m")),
            "near_collision_count": int(_float_or_none(aggregate.get("near_collision_count")) or 0),
            "collision_count": int(_float_or_none(aggregate.get("collision_count")) or 0),
            "planning_failures": int(_float_or_none(aggregate.get("planning_failures")) or 0),
            "localisation_error_mean_m": _float_or_none(aggregate.get("localisation_error_mean_m")),
            "localisation_error_max_m": _float_or_none(aggregate.get("localisation_error_max_m")),
            "goal_distance_gt_m": assessment["goal_distance_gt_m"],
        },
        "provenance": {
            "container_digest": aggregate.get("container_digest"),
            "host": aggregate.get("host"),
            "gpu": aggregate.get("gpu"),
            "ros_distro": aggregate.get("ros_distro"),
            "gazebo_version": aggregate.get("gazebo_version"),
            "ros_domain_id": None,
            "gz_partition": None,
            "bag_path": str(bag_dir),
            "bag_mcap_count": 1,
            "bag_checksum_sha256": bag_checksum_sha256,
            "recording_profile": RECORDING_PROFILE,
            "topic_health_sidecar": str(topic_health_sidecar),
            "event_sidecar": str(event_sidecar),
            "event_source": EVENT_SOURCE,
            "time_base": TIME_BASE,
            "receive_clock_map": str(clock_map_path),
            "receive_clock_map_sha256": clock_map_sha256,
            "research1": {
                "run_id": run_id,
                "bag_metadata_sha256": catalog_row.get("metadata_sha256"),
                "aggregate_path": str(aggregate_path),
                "aggregate_sha256": aggregate_sha256,
                "catalog_admission_status": catalog_row.get("admission_status"),
                "adapter_version": ADAPTER_VERSION,
            },
            "protected_test_used": False,
        },
    }


def load_receive_clock_map(summary: Mapping[str, Any]) -> ReceiveClockMap | None:
    """Return the declared clock map of an adapted summary, or None for native episodes."""
    provenance = summary.get("provenance", {}) if isinstance(summary, Mapping) else {}
    path = provenance.get("receive_clock_map")
    if not path:
        return None
    if provenance.get("time_base") != TIME_BASE:
        raise ValueError("summary declares a receive clock map with an unexpected time base")
    clock_path = Path(str(path))
    expected = provenance.get("receive_clock_map_sha256")
    if expected and sha256_path(clock_path) != expected:
        raise ValueError(f"receive clock map checksum differs: {clock_path}")
    return ReceiveClockMap.from_json(json.loads(clock_path.read_text(encoding="utf-8")))


# ------------------------------------------------------------- supplement sizing


PROTOCOL_FITTING_TARGET = 3000
BALANCED_DEVELOPMENT_EPISODES = 648
TARGETED_DEVELOPMENT_EPISODES = 1212
MINIMUM_SUPPLEMENT_EPISODES = 324
ROUTES_PER_BLOCK = 12


def supplement_episode_count(admitted_research1_development: int) -> int:
    """max(324, 3000 - 648 - 1212 - N_r1), rounded up to complete 12-route blocks."""
    if admitted_research1_development < 0:
        raise ValueError("admitted count must be non-negative")
    shortfall = (
        PROTOCOL_FITTING_TARGET - BALANCED_DEVELOPMENT_EPISODES
        - TARGETED_DEVELOPMENT_EPISODES - int(admitted_research1_development)
    )
    episodes = max(MINIMUM_SUPPLEMENT_EPISODES, shortfall)
    return math.ceil(episodes / ROUTES_PER_BLOCK) * ROUTES_PER_BLOCK


def distribute_blocks(blocks: int, conditions: int) -> list[int]:
    """Spread replicate blocks over conditions so counts differ by at most one."""
    if conditions <= 0 or blocks < 0:
        raise ValueError("blocks must be non-negative and conditions positive")
    base, extra = divmod(blocks, conditions)
    return [base + (1 if index < extra else 0) for index in range(conditions)]


def assessment_to_manifest_row(catalog_row: Mapping[str, Any], assessment: Mapping[str, Any],
                               *, aggregate_path: Path, aggregate_sha256: str) -> dict[str, Any]:
    row = {**catalog_row}
    row.update({
        key: value for key, value in assessment.items()
        if key not in {"reasons", "admissible", "run_id"}
    })
    row.update({
        "episode_key": episode_key(str(catalog_row["run_id"])),
        "campaign_id": CAMPAIGN_ID,
        "aggregate_path": str(aggregate_path),
        "aggregate_sha256": aggregate_sha256,
        "adapter_version": ADAPTER_VERSION,
        "admission_status": "causal_adapter_admitted_pending_human_gate",
        "protected_outcomes_consulted": False,
        "protected_test_used": False,
    })
    return row


__all__ = [
    "ADAPTER_VERSION", "CAMPAIGN_ID", "RESEARCH1_ROOT", "TIME_BASE", "EVENT_SOURCE",
    "REQUIRED_FEATURE_TOPICS", "OPTIONAL_FEATURE_TOPICS", "LABEL_EVIDENCE_TOPICS",
    "RESEARCH2_FAULT_FAMILIES", "TERMINAL_EVENT_CLASS", "EVENT_TIME_DERIVATION",
    "EPISODE_START_DEFINITION", "AdapterThresholds", "RouteSpec", "BagEvidence",
    "ReceiveClockMap", "research2_fault_family", "research2_severity", "episode_key",
    "index_aggregates", "load_aggregate", "load_route_spec", "read_bag_evidence",
    "build_pose_error_samples", "build_motion_samples", "assess_episode",
    "build_event_sidecar", "build_summary", "load_receive_clock_map",
    "supplement_episode_count", "distribute_blocks", "assessment_to_manifest_row",
    "sha256_path", "sha256_bytes", "asdict",
]

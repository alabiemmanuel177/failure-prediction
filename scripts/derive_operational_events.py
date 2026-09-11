#!/usr/bin/env python3
"""Derive review-pending operational terminal events from one immutable MCAP.

The output is label-only evidence. It never edits an annotation, summary, or raw bag;
an independent reviewer must confirm any proposed event before it enters causal labels.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import json
import math
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.labels.operational_events import (
    MotionSample,
    PoseErrorSample,
    first_immobilisation,
    first_localisation_loss,
    first_primary_event,
)
from src.protected_data import enforce_protected_boundary
from src.research1_adapter import load_receive_clock_map


TERMINAL_CLASS = {
    "collision": "collision",
    "planner_failure": "navigation_abort",
    "false_arrival": "false_arrival",
    "timeout": "mission_timeout",
}


def stamp_s(message) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def yaw(orientation) -> float:
    return math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )


def yaw_error(left: float, right: float) -> float:
    return abs(math.atan2(math.sin(left - right), math.cos(left - right)))


def nearest_pose(
    timestamp: float,
    timestamps: list[float],
    poses: list[tuple[float, float, float]],
    maximum_gap_seconds: float = 0.5,
) -> tuple[float, float, float] | None:
    index = bisect_left(timestamps, timestamp)
    candidates = [candidate for candidate in (index - 1, index, index + 1)
                  if 0 <= candidate < len(timestamps)]
    if not candidates:
        return None
    selected = min(candidates, key=lambda candidate: abs(timestamps[candidate] - timestamp))
    if abs(timestamps[selected] - timestamp) > maximum_gap_seconds:
        return None
    return poses[selected]


def read_evidence(
    bag: Path, receive_clock=None,
) -> tuple[list[PoseErrorSample], list[MotionSample], dict]:
    """Read label evidence. ``receive_clock`` maps recorder receive seconds onto the
    simulation clock for adapted Research 1 bags (wall-clock recorder); native
    Research 2 bags were recorded on simulation time and pass None."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    required = {"/ground_truth_pose", "/amcl_pose", "/cmd_vel", "/odom"}
    missing = sorted(required - set(types))
    if missing:
        raise ValueError(f"bag lacks label-evidence topics: {missing}")
    ground_truth: list[tuple[float, float, float, float]] = []
    amcl: list[tuple[float, float, float, float]] = []
    commands: list[tuple[float, float]] = []
    odometry: list[tuple[float, float, float]] = []
    selected = required
    while reader.has_next():
        topic, data, receive_timestamp_ns = reader.read_next()
        if topic not in selected:
            continue
        message = deserialize_message(data, get_message(types[topic]))
        if topic == "/ground_truth_pose":
            ground_truth.append((
                stamp_s(message), float(message.pose.position.x),
                float(message.pose.position.y), yaw(message.pose.orientation),
            ))
        elif topic == "/amcl_pose":
            amcl.append((
                stamp_s(message), float(message.pose.pose.position.x),
                float(message.pose.pose.position.y), yaw(message.pose.pose.orientation),
            ))
        elif topic == "/cmd_vel":
            receive_seconds = receive_timestamp_ns * 1e-9
            if receive_clock is not None:
                receive_seconds = receive_clock.to_sim(receive_seconds)
            commands.append((receive_seconds, float(message.linear.x)))
        elif topic == "/odom":
            odometry.append((
                stamp_s(message), float(message.pose.pose.position.x),
                float(message.pose.pose.position.y),
            ))

    ground_truth.sort(); amcl.sort(); commands.sort(); odometry.sort()
    gt_times = [item[0] for item in ground_truth]
    gt_poses = [(item[1], item[2], item[3]) for item in ground_truth]
    pose_errors: list[PoseErrorSample] = []
    unmatched_amcl = 0
    for timestamp, x, y, estimate_yaw in amcl:
        truth = nearest_pose(timestamp, gt_times, gt_poses)
        if truth is None:
            unmatched_amcl += 1
            continue
        pose_errors.append(PoseErrorSample(
            timestamp=timestamp,
            translation_error_m=math.dist((x, y), truth[:2]),
            yaw_error_rad=yaw_error(estimate_yaw, truth[2]),
        ))

    command_times = [item[0] for item in commands]
    motion: list[MotionSample] = []
    odom_without_command = 0
    for timestamp, x, y in odometry:
        index = bisect_right(command_times, timestamp) - 1
        if index < 0:
            odom_without_command += 1
            continue
        command_time, linear = commands[index]
        if timestamp - command_time > 0.5:
            odom_without_command += 1
            continue
        motion.append(MotionSample(timestamp, linear, x, y))
    integrity = {
        "ground_truth_samples": len(ground_truth),
        "amcl_samples": len(amcl),
        "matched_pose_error_samples": len(pose_errors),
        "unmatched_amcl_samples": unmatched_amcl,
        "command_samples": len(commands),
        "odometry_samples": len(odometry),
        "matched_motion_samples": len(motion),
        "odometry_without_fresh_command": odom_without_command,
    }
    return pose_errors, motion, integrity


def event_time(event: dict) -> float:
    stamp = event["stamp"]
    return float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-protected-after-freeze", action="store_true")
    args = parser.parse_args()
    summary = yaml.safe_load(args.summary.read_text(encoding="utf-8"))
    protected = summary.get("environment", {}).get("protected_test_used")
    gate_passed = False
    if protected is True and args.allow_protected_after_freeze:
        gate_passed = subprocess.run([
            sys.executable, str(ROOT / "scripts/check_readiness.py"),
            "--stage", "confirmatory",
        ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    try:
        enforce_protected_boundary(
            protected,
            explicitly_allowed=args.allow_protected_after_freeze,
            confirmatory_gate_passed=gate_passed,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    bag = Path(summary["provenance"]["bag_path"])
    try:
        receive_clock = load_receive_clock_map(summary)
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"receive clock map is invalid: {error}") from error
    pose_errors, motion, integrity = read_evidence(bag, receive_clock)
    if receive_clock is not None:
        integrity["time_base"] = str(summary["provenance"].get("time_base"))
        integrity["receive_clock_map"] = str(summary["provenance"].get("receive_clock_map"))
    config = yaml.safe_load((ROOT / "configs/failure_events.yaml").read_text())
    localisation = config["events"]["localisation_loss"]
    immobilisation = config["events"]["immobilisation"]
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
    sidecar = Path(summary["provenance"]["event_sidecar"])
    events = json.loads(sidecar.read_text(encoding="utf-8"))
    terminal = next((item for item in events if item.get("event_type") == "terminal_event"), None)
    if terminal is None:
        raise SystemExit("event sidecar lacks terminal_event")
    terminal_state = str(terminal.get("parameters", {}).get(
        "terminal_state", summary["outcome"]["terminal_state"]
    ))
    terminal_class = TERMINAL_CLASS.get(terminal_state)
    if terminal_class:
        candidates[terminal_class] = event_time(terminal)
    primary = first_primary_event(candidates, config["event_precedence"])
    output = {
        "schema_version": 1,
        "protocol_version": summary["identity"]["protocol_version"],
        "status": "automatic_review_pending",
        "run_id": summary["identity"]["run_id"],
        "protected_test_used": protected,
        "evidence_source": {
            "bag_path": str(bag),
            "bag_checksum_sha256": summary["provenance"]["bag_checksum_sha256"],
            "event_sidecar": str(sidecar),
        },
        "integrity": integrity,
        "candidate_event_times": candidates,
        "primary_candidate": (
            {"class": primary[0], "time": primary[1]} if primary else None
        ),
        "limitations": [
            "unsafe_perception requires explicit obstacle-miss and emergency-intervention evidence and is not inferred from fault identity",
            "automatic candidates require independent timeline review before causal-label admission",
        ],
    }
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, sort_keys=False)
    print(f"wrote review-pending operational events to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

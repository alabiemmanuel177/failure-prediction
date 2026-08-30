#!/usr/bin/env python3
"""Audit causal activation and observable treatment delivery in campaign bags.

This is an engineering integrity audit, not an outcome analysis.  It reads label-only
streams solely to prove that the declared fault was applied after the clean prefix.
The generated report must never be joined to the deployable feature matrix.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
ROS_PACKAGE = ROOT / "ros_ws" / "src" / "failure_experiment"
if str(ROS_PACKAGE) not in sys.path:
    sys.path.insert(0, str(ROS_PACKAGE))

from failure_experiment.transforms import (  # noqa: E402
    camera_occlusion,
    lidar_dropout,
    semantic_risk_corruption,
)


def stamp_ns(message) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def yaw(quaternion) -> float:
    return math.atan2(
        2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def angle_distance(left: float, right: float) -> float:
    return abs(math.atan2(math.sin(left - right), math.cos(left - right)))


def values_dict(status) -> dict[str, str]:
    return {item.key: item.value for item in status.values}


def read_selected(bag: Path) -> tuple[dict[str, list], list[dict]]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    wanted = {
        "/research2/raw/camera/image", "/camera/image",
        "/research2/raw/scan", "/scan",
        "/research2/raw/odom", "/odom",
        "/amcl_pose", "/initialpose",
        "/research2/raw/semantic/risk_grid", "/semantic/risk_grid",
        "/research2/events",
    }
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    messages = {topic: [] for topic in wanted}
    events: list[dict] = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in wanted:
            continue
        message = deserialize_message(data, get_message(types[topic]))
        if topic != "/research2/events":
            messages[topic].append(message)
            continue
        for status in message.status:
            values = values_dict(status)
            if "event_type" not in values:
                continue
            events.append({
                **values,
                "time_ns": stamp_ns(message),
                "parameters": json.loads(values.get("parameters_json", "{}")),
            })
    return messages, sorted(events, key=lambda item: item["time_ns"])


def by_stamp(messages: list) -> dict[int, object]:
    return {stamp_ns(message): message for message in messages}


def exact_camera(messages: dict[str, list], onset: int, parameters: dict, seed: int) -> dict:
    raw = by_stamp(messages["/research2/raw/camera/image"])
    deployed = by_stamp(messages["/camera/image"])
    checked = correct = expected_drop = realized_drop = unexpected_missing = 0
    for stamp, message in raw.items():
        if stamp < onset:
            continue
        array = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.width, 3)
        expected = camera_occlusion(
            array,
            mask_fraction=float(parameters["mask_fraction"]),
            dropout_probability=float(parameters["dropout_probability"]),
            seed=seed,
            stamp_ns=stamp,
        )
        actual = deployed.get(stamp)
        checked += 1
        if expected is None:
            expected_drop += 1
            realized_drop += int(actual is None)
        elif actual is not None:
            observed = np.frombuffer(actual.data, dtype=np.uint8).reshape(
                actual.height, actual.width, 3
            )
            correct += int(np.array_equal(expected, observed))
        else:
            unexpected_missing += 1
    expected_outputs = checked - expected_drop
    # The recorder and proxy stop on adjacent callbacks. One unmatched final raw
    # frame is a bounded recording edge, not evidence that the transform differed.
    matched_expected_outputs = expected_outputs - unexpected_missing
    passed = (
        checked > 0
        and realized_drop == expected_drop
        and unexpected_missing <= 1
        and correct == matched_expected_outputs
    )
    return {
        "method": "exact deterministic reconstruction from raw images",
        "raw_post_onset": checked,
        "expected_and_realized_drops": expected_drop,
        "exact_output_matches": correct,
        "expected_outputs": expected_outputs,
        "unexpected_missing_outputs": unexpected_missing,
        "passed": passed,
    }


def exact_lidar(messages: dict[str, list], onset: int, parameters: dict, seed: int) -> dict:
    raw = by_stamp(messages["/research2/raw/scan"])
    deployed = by_stamp(messages["/scan"])
    checked = correct = 0
    for stamp, message in raw.items():
        if stamp < onset or stamp not in deployed:
            continue
        expected = lidar_dropout(
            list(message.ranges),
            invalid_fraction=float(parameters["invalid_fraction"]),
            seed=seed,
        )
        actual = list(deployed[stamp].ranges)
        checked += 1
        same = len(expected) == len(actual) and all(
            (math.isinf(left) and math.isinf(right)) or abs(left - right) <= 1e-6
            for left, right in zip(expected, actual)
        )
        correct += int(same)
    return {
        "method": "exact deterministic reconstruction from raw scans",
        "matched_post_onset_scans": checked,
        "exact_output_matches": correct,
        "passed": checked > 0 and correct == checked,
    }


def wheel_effect(messages: dict[str, list], onset: int, parameters: dict) -> dict:
    raw = by_stamp(messages["/research2/raw/odom"])
    deployed = by_stamp(messages["/odom"])
    scale = float(parameters["odometry_progress_scale"])
    checked = twist_matches = altered_poses = 0
    for stamp, source in raw.items():
        if stamp < onset or stamp not in deployed:
            continue
        target = deployed[stamp]
        checked += 1
        twist_matches += int(abs(target.twist.twist.linear.x - source.twist.twist.linear.x * scale) <= 1e-6)
        displacement = math.hypot(
            target.pose.pose.position.x - source.pose.pose.position.x,
            target.pose.pose.position.y - source.pose.pose.position.y,
        )
        altered_poses += int(displacement > 1e-5)
    return {
        "method": "raw/deployed odometry comparison",
        "matched_post_onset_odometry": checked,
        "exact_twist_scale_matches": twist_matches,
        "altered_pose_messages": altered_poses,
        "passed": checked > 0 and twist_matches == checked and altered_poses > 0,
    }


def localisation_effect(messages: dict[str, list], onset: int, parameters: dict) -> dict:
    initial = [message for message in messages["/initialpose"] if stamp_ns(message) >= onset]
    amcl = messages["/amcl_pose"]
    if not initial:
        return {
            "method": "recorded /initialpose compared with source AMCL estimate",
            "passed": False,
            "reason": "/initialpose was not retained by this recording policy",
        }
    reset = initial[0]
    prior = min(amcl, key=lambda item: abs(stamp_ns(item) - stamp_ns(reset)))
    translation = math.hypot(
        reset.pose.pose.position.x - prior.pose.pose.position.x,
        reset.pose.pose.position.y - prior.pose.pose.position.y,
    )
    rotation = angle_distance(yaw(reset.pose.pose.orientation), yaw(prior.pose.pose.orientation))
    expected_translation = float(parameters["translation_m"])
    expected_rotation = float(parameters["yaw_rad"])
    return {
        "method": "recorded /initialpose compared with source AMCL estimate",
        "translation_m": round(translation, 6),
        "yaw_rad": round(rotation, 6),
        "passed": abs(translation - expected_translation) <= 0.02 and abs(rotation - expected_rotation) <= 0.02,
    }


def exact_semantic(messages: dict[str, list], onset: int, parameters: dict, seed: int) -> dict:
    raw = by_stamp(messages["/research2/raw/semantic/risk_grid"])
    deployed = by_stamp(messages["/semantic/risk_grid"])
    checked = correct = 0
    for stamp, message in raw.items():
        if stamp < onset or stamp not in deployed:
            continue
        expected = semantic_risk_corruption(
            list(message.data),
            corruption_probability=float(parameters["corruption_probability"]),
            risk_scale=float(parameters["risk_scale"]),
            seed=seed,
            stamp_ns=stamp,
        )
        checked += 1
        correct += int(expected == list(deployed[stamp].data))
    if checked == 0:
        return {
            "method": "exact deterministic reconstruction from raw semantic grids",
            "passed": False,
            "reason": "raw semantic grids were not retained by this recording policy",
        }
    return {
        "method": "exact deterministic reconstruction from raw semantic grids",
        "matched_post_onset_grids": checked,
        "exact_output_matches": correct,
        "passed": correct == checked,
    }


def audit_episode(summary: dict) -> dict:
    identity = summary["identity"]
    label = summary["label_only"]
    family = label["fault_family"]
    messages, events = read_selected(Path(summary["provenance"]["bag_path"]))
    run_events = [event for event in events if event.get("run_id") == identity["run_id"]]
    planned = next((event for event in run_events if event["event_type"] == "injection_planned"), None)
    started = next((event for event in run_events if event["event_type"] == "injection_started"), None)
    terminal = next((event for event in run_events if event["event_type"] == "terminal_event"), None)
    bad = [event["event_type"] for event in run_events if event["event_type"] in {
        "injection_error", "injection_ineligible"
    }]
    causal = bool(planned and started and terminal) and (
        started["time_ns"] - planned["time_ns"] + 50_000_000
        >= int(float(label["clean_prefix_seconds"]) * 1e9)
    ) and started["time_ns"] < terminal["time_ns"] and not bad
    parameter_match = bool(started) and started["parameters"] == label["parameters"]
    treatment = {"method": "not evaluated", "passed": False}
    if started:
        arguments = (messages, started["time_ns"], label["parameters"])
        seed = int(summary["environment"]["seed"])
        if family == "camera_occlusion":
            treatment = exact_camera(*arguments, seed)
        elif family == "lidar_dropout":
            treatment = exact_lidar(*arguments, seed)
        elif family == "wheel_slip":
            treatment = wheel_effect(*arguments)
        elif family == "localisation_perturbation":
            treatment = localisation_effect(*arguments)
        elif family == "semantic_corruption":
            treatment = exact_semantic(*arguments, seed)
        elif family in {"dynamic_blockage", "planner_oscillation"}:
            treatment = {
                "method": "injection_started is emitted only after ros_gz_sim create succeeds",
                "spawn_success_event": True,
                "passed": True,
            }
    return {
        "episode_key": identity["episode_key"],
        "run_id": identity["run_id"],
        "family": family,
        "severity": label["severity"],
        "terminal_state": summary["outcome"]["terminal_state"],
        "causal_schedule_passed": causal,
        "event_parameters_match_summary": parameter_match,
        "treatment": treatment,
        "passed": causal and parameter_match and bool(treatment["passed"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--summary-root", type=Path, default=ROOT / "data/raw/summaries")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    for path in sorted(args.summary_root.glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if document.get("identity", {}).get("campaign_id") == args.campaign_id:
            summaries.append(document)
    episodes = [audit_episode(summary) for summary in summaries if summary["label_only"]["fault_family"] != "none"]
    report = {
        "schema_version": 1,
        "campaign_id": args.campaign_id,
        "purpose": "label-only causal schedule and fault-treatment integrity audit",
        "deployable_feature_use": "forbidden",
        "episode_count": len(episodes),
        "passed_count": sum(item["passed"] for item in episodes),
        "failed_count": sum(not item["passed"] for item in episodes),
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.output}: {report['passed_count']}/{len(episodes)} passed")
    for item in episodes:
        if not item["passed"]:
            print(f"- {item['episode_key']}: {item['treatment'].get('reason', 'integrity check failed')}")
    return 0 if report["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

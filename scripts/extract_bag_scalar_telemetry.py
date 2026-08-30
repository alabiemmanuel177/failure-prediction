#!/usr/bin/env python3
"""Extract causal-ready scalar telemetry from one retained Research 2 MCAP."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.derived import covariance_trace, float_image_mean, path_summaries, scan_summaries


MAX_AGE = {
    "/cmd_vel": 0.5, "/odom": 0.5, "/amcl_pose": 2.0, "/scan": 0.5,
    "/plan": 2.0, "/local_plan": 1.0, "/semantic/confidence": 1.0,
    "/semantic/uncertainty": 1.0, "/semantic/inference_latency_ms": 1.0,
}


def feature_values(topic: str, message) -> dict[str, float]:
    if topic == "/cmd_vel":
        return {"command_linear": float(message.linear.x),
                "command_angular": float(message.angular.z)}
    if topic == "/odom":
        return {"measured_linear": float(message.twist.twist.linear.x),
                "measured_angular": float(message.twist.twist.angular.z),
                "odom_x": float(message.pose.pose.position.x),
                "odom_y": float(message.pose.pose.position.y)}
    if topic == "/amcl_pose":
        return {"pose_covariance_trace": covariance_trace(message.pose.covariance),
                "amcl_x": float(message.pose.pose.position.x),
                "amcl_y": float(message.pose.pose.position.y)}
    if topic == "/scan":
        return scan_summaries(message.ranges, float(message.angle_min), float(message.angle_increment))
    if topic in {"/plan", "/local_plan"}:
        points = [(float(pose.pose.position.x), float(pose.pose.position.y))
                  for pose in message.poses]
        prefix = "global" if topic == "/plan" else "local"
        return {f"{prefix}_{name}": value for name, value in path_summaries(points).items()}
    if topic == "/semantic/confidence":
        return {"confidence_mean": float_image_mean(message)}
    if topic == "/semantic/uncertainty":
        return {"uncertainty_mean": float_image_mean(message)}
    if topic == "/semantic/inference_latency_ms":
        return {"inference_latency_ms": float_image_mean(message)}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("run_id")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    selected = set(MAX_AGE) & set(types)
    rows = []
    while reader.has_next():
        topic, data, receive_timestamp_ns = reader.read_next()
        if topic not in selected:
            continue
        message = deserialize_message(data, get_message(types[topic]))
        for feature, value in feature_values(topic, message).items():
            rows.append({
                "run_id": args.run_id,
                # Recorder receive time is the availability time. Header time alone can
                # make a late-arriving message look causally available too early.
                "timestamp": receive_timestamp_ns * 1e-9,
                "feature": feature, "source": topic, "value": value,
                "max_age_seconds": MAX_AGE[topic],
            })
    if not rows:
        raise SystemExit("no supported scalar telemetry found in bag")
    rows.sort(key=lambda row: (row["feature"], float(row["timestamp"])))
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} scalar observations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

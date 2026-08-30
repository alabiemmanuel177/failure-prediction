#!/usr/bin/env python3
"""Validate one Research 2 summary, MCAP metadata, and topic-health sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


CORE_TOPICS = {
    "/clock",
    "/research2/events",
    "/diagnostics",
    "/research2/raw/camera/image",
    "/research2/raw/scan",
    "/research2/raw/odom",
    "/camera/image",
    "/scan",
    "/odom",
    "/cmd_vel",
    "/behavior_tree_log",
    "/ground_truth_pose",
}
SEMANTIC_TOPICS = {
    "/semantic/classes",
    "/semantic/confidence",
    "/semantic/uncertainty",
    "/semantic/risk_grid",
    "/semantic/risk_grid_odom",
}
SEMANTIC_AUDIT_TOPICS = {
    "/research2/raw/semantic/risk_grid",
    "/research2/raw/semantic/risk_grid_odom",
}


def recording_window_counts(
    bag_dir: Path, selected: set[str]
) -> tuple[bool, dict[str, int], list[dict[str, object]]]:
    """Count selected messages at/after the retained recording-window marker."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    marker_timestamp = None
    window_ended = False
    counts = {topic: 0 for topic in selected}
    events: list[dict[str, object]] = []
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic == "/research2/events":
            message = deserialize_message(data, get_message(types[topic]))
            for status in message.status:
                values = {item.key: item.value for item in status.values}
                if "event_type" in values:
                    events.append({
                        "event_type": values["event_type"],
                        "stamp_seconds": (
                            float(message.header.stamp.sec)
                            + float(message.header.stamp.nanosec) * 1e-9
                        ),
                    })
                if values.get("event_type") == "recording_window_started":
                    marker_timestamp = timestamp
                    window_ended = False
                    counts = {name: 0 for name in selected}
                    break
                if values.get("event_type") == "recording_window_ended":
                    window_ended = True
                    break
        if (
            marker_timestamp is not None
            and not window_ended
            and timestamp >= marker_timestamp
            and topic in counts
        ):
            counts[topic] += 1
    return marker_timestamp is not None, counts, events


def raw_deployed_tolerance(
    *, raw_topic: str, raw_count: int, summary: dict
) -> int:
    """Bound edge mismatch plus declared camera frame dropout, and nothing else."""
    edge = max(5, int(0.05 * raw_count))
    label = summary.get("label_only", {})
    if raw_topic == "/research2/raw/camera/image" and label.get("fault_family") == "camera_occlusion":
        probability = float(label.get("parameters", {}).get("dropout_probability", 0.0))
        # Ten percentage points cover finite-sample variation in this small integrity
        # campaign; treatment validation separately measures the realized fraction.
        return max(edge, int((probability + 0.10) * raw_count))
    return edge


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    summary = yaml.safe_load(args.summary.read_text(encoding="utf-8"))
    findings: list[str] = []
    run_id = summary["identity"]["run_id"]
    environment = summary["environment"]
    if environment.get("split") not in {"development", "validation"}:
        findings.append("episode is not in an allowed pre-freeze split")
    if environment.get("protected_test_used") is not False:
        findings.append("protected_test_used must be false")

    bag_dir = Path(summary["provenance"]["bag_path"])
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.exists():
        findings.append(f"missing bag metadata: {metadata_path}")
        metadata = {}
    else:
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")).get(
            "rosbag2_bagfile_information", {}
        )
    custom = metadata.get("custom_data", {})
    if str(custom.get("run_id")) != run_id:
        findings.append("bag custom_data run_id does not match summary")
    identity = summary["identity"]
    platform_commit = identity.get("research1_platform_commit", identity.get("research1_commit"))
    custom_platform = custom.get("research1_platform_commit", custom.get("research1_commit"))
    if str(custom_platform) != str(platform_commit):
        findings.append("bag Research 1 platform commit does not match summary")
    if identity.get("research1_repository_head") is not None and str(
        custom.get("research1_repository_head")
    ) != str(identity["research1_repository_head"]):
        findings.append("bag Research 1 repository HEAD does not match summary")

    topic_counts = {
        item["topic_metadata"]["name"]: int(item["message_count"])
        for item in metadata.get("topics_with_message_count", [])
    }
    required = set(CORE_TOPICS)
    if environment.get("system_id") != "S0":
        required.update(SEMANTIC_TOPICS)
    # Raw semantic grids and /initialpose were added after the definitive v3
    # campaign. Require them for new treatment-verification supplements while
    # keeping the immutable v3 bags valid under their original recording policy.
    if identity.get("campaign_id", "").startswith("treatment_verification_"):
        required.add("/initialpose")
        if summary.get("label_only", {}).get("fault_family") == "semantic_corruption":
            required.update(SEMANTIC_AUDIT_TOPICS)
    for topic in sorted(required):
        if topic_counts.get(topic, 0) <= 0:
            findings.append(f"required topic has no recorded messages: {topic}")
    if topic_counts.get("/research2/events", 0) < 2:
        findings.append("fewer than two research events were recorded")

    for raw, deployed in (
        ("/research2/raw/camera/image", "/camera/image"),
        ("/research2/raw/scan", "/scan"),
        ("/research2/raw/odom", "/odom"),
    ):
        raw_count, deployed_count = topic_counts.get(raw, 0), topic_counts.get(deployed, 0)
        tolerance = raw_deployed_tolerance(
            raw_topic=raw, raw_count=raw_count, summary=summary
        )
        if raw_count and abs(raw_count - deployed_count) > tolerance:
            findings.append(
                f"raw/deployed count mismatch exceeds tolerance: {raw}={raw_count}, "
                f"{deployed}={deployed_count}"
            )

    health_path = Path(summary["provenance"]["topic_health_sidecar"])
    if not health_path.exists():
        findings.append(f"missing topic-health sidecar: {health_path}")
    else:
        health = json.loads(health_path.read_text(encoding="utf-8"))
        if health.get("run_id") != run_id:
            findings.append("topic-health run_id does not match summary")
        marker_found, window_counts, recorded_events = recording_window_counts(
            bag_dir, {"/camera/image", "/scan", "/odom"}
        )
        if not marker_found:
            findings.append("recording-window marker is absent from MCAP")
        if health.get("recording_window_started") is not True:
            findings.append("topic-health did not observe the recording-window marker")
        if health.get("recording_window_ended") is not True:
            findings.append("topic-health did not observe the recording-window end marker")
        for topic in ("/camera/image", "/scan", "/odom"):
            observed = int(health.get("counts", {}).get(topic, 0))
            recorded = window_counts.get(topic, 0)
            # The independent subscriber and recorder start/stop at slightly different
            # instants. Beyond that bounded edge allowance, fewer bag messages are
            # evidence of recorder loss and fewer health messages invalidate the audit.
            tolerance = max(5, int(0.05 * max(observed, recorded)))
            if abs(observed - recorded) > tolerance:
                findings.append(
                    f"topic-health/MCAP count mismatch exceeds tolerance: "
                    f"{topic} health={observed}, bag={recorded}"
                )

        family = summary.get("label_only", {}).get("fault_family", "none")
        if family != "none":
            event_types = [str(item["event_type"]) for item in recorded_events]
            if "injection_planned" not in event_types:
                findings.append("faulted episode has no injection_planned event")
            if "injection_started" not in event_types:
                findings.append("faulted episode has no injection_started event")
            if "injection_planned" in event_types and "injection_started" in event_types:
                planned = next(
                    float(item["stamp_seconds"])
                    for item in recorded_events if item["event_type"] == "injection_planned"
                )
                started = next(
                    float(item["stamp_seconds"])
                    for item in recorded_events if item["event_type"] == "injection_started"
                )
                clean_prefix = float(summary["label_only"]["clean_prefix_seconds"])
                if started - planned + 0.05 < clean_prefix:
                    findings.append("injection_started before the declared clean prefix elapsed")

    if findings:
        print(f"INVALID: {len(findings)} finding(s)")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(
        f"VALID: run={run_id} messages={metadata.get('message_count', 0)} "
        f"duration_s={metadata.get('duration', {}).get('nanoseconds', 0) / 1e9:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

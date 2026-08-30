#!/usr/bin/env python3
"""Create a review-pending annotation from a Research 2 summary and event bag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


CONFIRMATION_SOURCES = {
    "collision": ["/collision_event", "/cmd_vel"],
    "navigation_abort": ["/research2/events", "/behavior_tree_log"],
    "false_arrival": ["/research2/events", "/ground_truth_pose"],
    "localisation_loss": ["/ground_truth_pose", "/amcl_pose"],
    "immobilisation": ["/cmd_vel", "/odom"],
    "unsafe_perception": ["/camera/segmentation", "/research2/events"],
    "mission_timeout": ["/research2/events", "/clock"],
}


def values_dict(status) -> dict[str, str]:
    return {value.key: value.value for value in status.values}


def read_events(bag_path: Path) -> list[dict]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    events = []
    while reader.has_next():
        topic, data, _received_time = reader.read_next()
        if topic != "/research2/events":
            continue
        message = deserialize_message(data, get_message(types[topic]))
        for status in message.status:
            values = values_dict(status)
            events.append({
                **values,
                "time": message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
                "parameters": json.loads(values.get("parameters_json", "{}")),
            })
    return sorted(events, key=lambda event: event["time"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = yaml.safe_load(args.summary.read_text(encoding="utf-8"))
    events = read_events(Path(summary["provenance"]["bag_path"]))
    run_id = summary["identity"]["run_id"]
    run_events = [event for event in events if event.get("run_id") == run_id]
    goal = next((event for event in run_events if event["event_type"] == "goal_dispatched"), None)
    terminal = next((event for event in run_events if event["event_type"] == "terminal_event"), None)
    if goal is None or terminal is None:
        raise SystemExit("bag lacks goal_dispatched or terminal_event")
    started = next(
        (event for event in run_events if event["event_type"] == "injection_started"), None
    )
    ended = next(
        (event for event in run_events if event["event_type"] == "injection_ended"), None
    )
    ineligible = next(
        (event for event in run_events if event["event_type"] == "injection_ineligible"),
        None,
    )
    label = summary["label_only"]
    primary_class = label.get("primary_event_class")
    event_rows = []
    if primary_class:
        event_rows.append({
            "event_id": "event-001",
            "class": primary_class,
            "time": terminal["time"],
            "terminal": True,
            "confirmation_sources": CONFIRMATION_SOURCES[primary_class],
            "confidence": "confirmed",
            "notes": "Automatically extracted; requires human timeline review.",
        })
    annotation = {
        "schema_version": 1,
        "taxonomy_version": "1.0.0",
        "protocol_version": summary["identity"]["protocol_version"],
        "run_id": run_id,
        "annotator": "automatic_event_extractor",
        "annotation_timestamp_utc": summary["identity"]["timestamp_utc"],
        "episode": {
            "start_time": goal["time"],
            "end_time": terminal["time"],
            "termination_reason": summary["outcome"]["terminal_state"],
        },
        "injections": [{
            "family": label["fault_family"],
            "severity": None if label["fault_family"] == "none" else label["severity"],
            "planned_onset": goal["time"] + float(label["planned_onset_seconds"]),
            "actual_onset": started["time"] if started else None,
            "duration_seconds": (
                ended["time"] - started["time"] if started and ended else None
            ),
            "eligible": started is not None and ineligible is None,
        }],
        "events": event_rows,
        "episode_exclusion": {"excluded": False, "reason": None, "notes": None},
        "review": {
            "second_reviewer": None,
            "adjudication_status": "pending",
            "disagreements": [],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(annotation, stream, sort_keys=False)
    print(f"wrote review-pending annotation to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


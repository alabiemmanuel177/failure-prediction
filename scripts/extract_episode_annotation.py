#!/usr/bin/env python3
"""Create a review-pending annotation from a Research 2 summary and event bag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.protected_data import enforce_protected_boundary  # noqa: E402


CONFIRMATION_SOURCES = {
    "collision": ["/collision_event", "/cmd_vel"],
    "navigation_abort": ["/research2/events", "/behavior_tree_log"],
    "false_arrival": ["/research2/events", "/ground_truth_pose"],
    "localisation_loss": ["/ground_truth_pose", "/amcl_pose"],
    "immobilisation": ["/cmd_vel", "/odom"],
    "unsafe_perception": ["/camera/segmentation", "/research2/events"],
    "mission_timeout": ["/research2/events", "/clock"],
}


def resolve_primary_event(
    summary: dict, terminal_event: dict, operational: dict | None,
) -> tuple[str | None, float | None]:
    """Resolve the first operational event, or the legacy terminal-only fallback."""
    if operational is None:
        event_class = summary["label_only"].get("primary_event_class")
        return event_class, float(terminal_event["time"]) if event_class else None
    if operational.get("run_id") != summary["identity"]["run_id"]:
        raise ValueError("operational-event evidence run_id differs from summary")
    if operational.get("protected_test_used") is not summary.get(
        "environment", {}
    ).get("protected_test_used"):
        raise ValueError("operational-event evidence protection marker differs from summary")
    if operational.get("status") != "automatic_review_pending":
        raise ValueError("operational-event evidence has an unexpected status")
    candidate = operational.get("primary_candidate")
    if candidate is None:
        return None, None
    if not isinstance(candidate, dict) or candidate.get("class") not in CONFIRMATION_SOURCES:
        raise ValueError("operational primary candidate is malformed or unknown")
    event_time = float(candidate["time"])
    terminal_time = float(terminal_event["time"])
    # The upper bound is checked against the actual terminal event here; annotation
    # validation checks the lower goal-dispatch bound after construction.
    if event_time > terminal_time:
        raise ValueError("operational primary candidate occurs after terminal event")
    return str(candidate["class"]), event_time


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


def read_sidecar_events(summary: dict) -> list[dict]:
    """Events for adapted Research 1 episodes, whose bags carry no /research2/events.

    Used only when the summary provenance declares ``event_source: adapter_sidecar``;
    native Research 2 episodes always read the recorded event stream from the bag.
    """
    sidecar = Path(summary["provenance"]["event_sidecar"])
    run_id = summary["identity"]["run_id"]
    events = []
    for item in json.loads(sidecar.read_text(encoding="utf-8")):
        stamp = item["stamp"]
        events.append({
            "event_type": item["event_type"],
            "run_id": run_id,
            "time": float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9,
            "parameters": dict(item.get("parameters", {})),
            "reason": item.get("reason", ""),
        })
    return sorted(events, key=lambda event: event["time"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--operational-events", type=Path)
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
    adapted_research1 = summary["provenance"].get("event_source") == "adapter_sidecar"
    if adapted_research1:
        if summary.get("label_only", {}).get("research1_adapted") is not True:
            raise SystemExit("adapter sidecar events require a research1_adapted summary")
        events = read_sidecar_events(summary)
    else:
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
    operational = (
        yaml.safe_load(args.operational_events.read_text(encoding="utf-8"))
        if args.operational_events else None
    )
    try:
        primary_class, primary_time = resolve_primary_event(summary, terminal, operational)
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    event_rows = []
    if primary_class:
        event_rows.append({
            "event_id": "event-001",
            "class": primary_class,
            "time": primary_time,
            "terminal": True,
            "confirmation_sources": CONFIRMATION_SOURCES[primary_class],
            "confidence": "confirmed",
            "notes": "Automatically extracted; requires human timeline review.",
        })
    if adapted_research1:
        # A Research 1 shift is a world-level condition present from before goal
        # dispatch, not a Research 2 injection with an onset: no injection guard applies
        # and the shift is recorded label-only beside the (empty) injection list.
        injections: list[dict] = []
    else:
        injections = [{
            "family": label["fault_family"],
            "severity": None if label["fault_family"] == "none" else label["severity"],
            "planned_onset": goal["time"] + float(label["planned_onset_seconds"]),
            "actual_onset": started["time"] if started else None,
            "duration_seconds": (
                ended["time"] - started["time"] if started and ended else None
            ),
            "eligible": started is not None and ineligible is None,
        }]
    annotation = {
        "schema_version": 1,
        "taxonomy_version": "1.0.0",
        "protocol_version": summary["identity"]["protocol_version"],
        "run_id": run_id,
        "protected_test_used": protected,
        "annotator": "automatic_event_extractor",
        "annotation_timestamp_utc": summary["identity"]["timestamp_utc"],
        "episode": {
            "start_time": goal["time"],
            "end_time": terminal["time"],
            "termination_reason": summary["outcome"]["terminal_state"],
        },
        "injections": injections,
        **({"research1_shift": {
            **dict(label.get("research1_shift", {})),
            "fault_family": label["fault_family"],
            "severity": label.get("severity"),
            "terminal_time_method": label.get("terminal_time_method"),
            "terminal_time_exactness": label.get("terminal_time_exactness"),
        }} if adapted_research1 else {}),
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

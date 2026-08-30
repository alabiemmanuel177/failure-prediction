"""Structured research events carried by standard ROS diagnostic messages."""

from __future__ import annotations

import json
from typing import Mapping

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue


EVENT_TOPIC = "/research2/events"


def event_message(*, stamp, event_type: str, run_id: str, family: str = "none",
                  severity: str = "none", seed: int = 0, eligible: bool = True,
                  reason: str = "", parameters: Mapping | None = None,
                  source_commit: str = "unknown") -> DiagnosticArray:
    values = {
        "event_type": event_type,
        "run_id": run_id,
        "family": family,
        "severity": severity,
        "seed": str(seed),
        "eligible": str(bool(eligible)).lower(),
        "reason": reason,
        "parameters_json": json.dumps(parameters or {}, sort_keys=True, separators=(",", ":")),
        "source_commit": source_commit,
        "causal_role": "label_only",
    }
    status = DiagnosticStatus()
    status.level = DiagnosticStatus.OK if eligible else DiagnosticStatus.WARN
    status.name = f"research2/{event_type}"
    status.message = reason or event_type
    status.hardware_id = "simulation"
    status.values = [KeyValue(key=key, value=value) for key, value in values.items()]
    message = DiagnosticArray()
    message.header.stamp = stamp
    message.header.frame_id = "research2_label_only"
    message.status = [status]
    return message


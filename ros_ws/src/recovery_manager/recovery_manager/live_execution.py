"""Guarded live recovery execution: gate, bounded action sequences, event logging.

UNTESTED LIVE. This module has never driven a simulator or robot. It is exercised only
through the fake commander in ``tests/test_recovery_live_execution.py``. It contains no
ROS imports so the decision logic stays unit-testable under the system interpreter.

Execution is possible only when the node parameter ``live_execution`` is true AND the
frozen evidence file (``configs/recovery_live_evidence.yaml``) records every item of
``required_live_evidence_before_execution`` from ``configs/recovery_guards.yaml`` as
verified with a timestamp against the current guard-config hash. Every guard
rejection and every executed action is reported through the ``log`` callback so the
node can publish them as ``/research2/events``-style diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol

import yaml


LIVE_EVIDENCE_DEFAULT = "configs/recovery_live_evidence.yaml"
# Every sequence begins with the safe-stop prefix (cancel the Nav2 task, then zero
# cmd_vel: "stop_command_and_nav2_cancel_order_validated"). A controlled stop resumes
# the mission afterwards: it is the default first response, not an abort. Only
# request_assistance ends the mission without an automated resume.
SAFE_STOP_PREFIX: tuple[str, ...] = ("cancel_task", "zero_cmd_vel")
ACTION_SEQUENCES: dict[str, tuple[str, ...]] = {
    "controlled_stop": ("cancel_task", "zero_cmd_vel", "resume_navigation"),
    "relocalise": ("cancel_task", "zero_cmd_vel", "relocalise", "resume_navigation"),
    "replan_clear_costmaps": ("cancel_task", "zero_cmd_vel", "clear_costmaps", "resume_navigation"),
    "backup": ("cancel_task", "zero_cmd_vel", "backup", "resume_navigation"),
    "spin_active_rescan": ("cancel_task", "zero_cmd_vel", "spin", "resume_navigation"),
    "wait": ("cancel_task", "zero_cmd_vel", "wait", "resume_navigation"),
    "request_assistance": ("cancel_task", "zero_cmd_vel", "request_assistance"),
}


@dataclass(frozen=True)
class ActionBounds:
    backup_distance_m: float = 0.25
    backup_speed_mps: float = 0.05
    spin_angle_rad: float = 1.5708
    wait_seconds: float = 3.0
    step_timeout_seconds: float = 10.0

    def validate(self) -> None:
        if not 0 < self.backup_distance_m <= 0.5:
            raise ValueError("backup distance must be bounded within (0, 0.5] m")
        if not 0 < self.backup_speed_mps <= 0.1:
            raise ValueError("backup speed must be bounded within (0, 0.1] m/s")
        if not 0 < self.spin_angle_rad <= 3.1416:
            raise ValueError("spin angle must be bounded within (0, pi] rad")
        if not 0 < self.wait_seconds <= 10.0 or self.step_timeout_seconds <= 0:
            raise ValueError("wait and step timeout must be positive and bounded")


@dataclass(frozen=True)
class GateDecision:
    enabled: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


def required_evidence_items(guard_config_path: Path) -> list[str]:
    document = yaml.safe_load(guard_config_path.read_text(encoding="utf-8"))
    items = document.get("required_live_evidence_before_execution") or []
    if len(items) != 5:
        raise ValueError("guard config must list the five required live-evidence items")
    return [str(item) for item in items]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp_ok(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def evaluate_live_gate(
    live_execution_parameter: bool, evidence_path: Path, guard_config_path: Path
) -> GateDecision:
    """Fail closed unless the parameter and the frozen evidence both authorise execution."""
    if not live_execution_parameter:
        return GateDecision(False, "live_execution parameter is false; recommendation-only mode")
    if not evidence_path.exists():
        return GateDecision(False, f"live evidence file is absent: {evidence_path}")
    try:
        evidence = yaml.safe_load(evidence_path.read_text(encoding="utf-8")) or {}
        required = required_evidence_items(guard_config_path)
    except (ValueError, yaml.YAMLError) as error:
        return GateDecision(False, f"live evidence unreadable: {error}")
    if not isinstance(evidence, dict) or evidence.get("frozen") is not True:
        return GateDecision(False, "live evidence is not frozen")
    if evidence.get("guard_config_sha256") != _sha256(guard_config_path):
        return GateDecision(False, "live evidence was frozen against a different guard config")
    items = evidence.get("required_live_evidence_before_execution") or {}
    for name in required:
        record = items.get(name) if isinstance(items, dict) else None
        if not isinstance(record, dict) or record.get("verified") is not True:
            return GateDecision(False, f"live evidence item not verified: {name}")
        if not _timestamp_ok(record.get("verified_utc")):
            return GateDecision(False, f"live evidence item lacks a timestamp: {name}")
    return GateDecision(True, "live execution authorised by frozen evidence", dict(evidence))


class Commander(Protocol):
    """Minimal interface the live executor drives; ``Nav2LiveCommander`` implements it."""

    def cancel_task(self) -> None: ...
    def zero_cmd_vel(self) -> None: ...
    def backup(self, distance_m: float, speed_mps: float) -> None: ...
    def spin(self, angle_rad: float) -> None: ...
    def wait(self, seconds: float) -> None: ...
    def clear_costmaps(self) -> None: ...
    def relocalise(self) -> None: ...
    def resume_navigation(self) -> None: ...
    def request_assistance(self) -> None: ...


class LiveExecutor:
    """Execute one guard-admitted decision as a bounded, fully logged action sequence."""

    def __init__(
        self, commander: Commander, gate: GateDecision, log: Callable[[dict[str, Any]], None],
        bounds: ActionBounds = ActionBounds(), clock: Callable[[], float] = time.monotonic,
    ) -> None:
        bounds.validate()
        self.commander = commander
        self.gate = gate
        self.log = log
        self.bounds = bounds
        self.clock = clock
        self.executed_count = 0

    def log_guard_rejections(self, decision: Mapping[str, Any]) -> list[str]:
        rejected = sorted(
            name for name, result in decision["guard_results"].items() if not result["eligible"]
        )
        self.log({
            "event_type": "recovery_guard_rejection",
            "run_id": decision["run_id"], "warning_id": decision["warning_id"],
            "policy_id": decision["policy_id"],
            "rejected_actions": rejected,
            "reasons": {name: decision["guard_results"][name]["reason"] for name in rejected},
            "recommended_action": decision["recommended_action"],
        })
        return rejected

    def safe_stop(self, run_id: str, warning_id: str, policy_id: str) -> dict[str, Any]:
        """Phase one of a live hand-off: cancel the task and zero cmd_vel, fully logged.

        The action-specific guards (relocalise, backup, spin) require a stopped robot,
        so the manager stops first, measures the stop from /odom, and only then decides.
        Without the live gate nothing moves and the refusal is logged.
        """
        record: dict[str, Any] = {
            "run_id": run_id, "warning_id": warning_id, "policy_id": policy_id,
            "phase": "safe_stop", "steps": [], "execution_performed": False,
        }
        if not self.gate.enabled:
            record["status"] = "refused_live_execution_locked"
            record["reason"] = self.gate.reason
            self.log({"event_type": "recovery_safe_stop", **record})
            return record
        started = self.clock()
        try:
            for step in SAFE_STOP_PREFIX:
                step_started = self.clock()
                self._perform(step)
                record["steps"].append({"step": step, "seconds": self.clock() - step_started})
            record["status"] = "executed"
            record["execution_performed"] = True
        except Exception as error:
            self.commander.zero_cmd_vel()
            record["status"] = "failed"
            record["reason"] = f"{type(error).__name__}: {error}"
        record["seconds"] = self.clock() - started
        self.log({"event_type": "recovery_safe_stop", **record})
        return record

    def execute(self, decision: Mapping[str, Any]) -> dict[str, Any]:
        rejected = self.log_guard_rejections(decision)
        action = str(decision["recommended_action"])
        record: dict[str, Any] = {
            "run_id": decision["run_id"], "warning_id": decision["warning_id"],
            "policy_id": decision["policy_id"], "action": action,
            "guard_rejected_actions": rejected, "steps": [], "execution_performed": False,
        }
        if not self.gate.enabled:
            record["status"] = "refused_live_execution_locked"
            record["reason"] = self.gate.reason
            self.log({"event_type": "recovery_execution_refused", **record})
            return record
        guard = decision["guard_results"].get(action, {"eligible": False, "reason": "unknown action"})
        if not guard["eligible"] or action not in ACTION_SEQUENCES:
            record["status"] = "refused_guard_rejected_action"
            record["reason"] = guard["reason"]
            self.log({"event_type": "recovery_execution_refused", **record})
            return record
        started = self.clock()
        try:
            for step in ACTION_SEQUENCES[action]:
                step_started = self.clock()
                self._perform(step)
                record["steps"].append({"step": step, "seconds": self.clock() - step_started})
            record["status"] = "executed"
            record["execution_performed"] = True
            self.executed_count += 1
        except Exception as error:  # never leave the robot moving after a failed step
            self.commander.zero_cmd_vel()
            record["status"] = "failed"
            record["reason"] = f"{type(error).__name__}: {error}"
            record["steps"].append({"step": "zero_cmd_vel_after_failure", "seconds": 0.0})
        record["seconds"] = self.clock() - started
        self.log({"event_type": "recovery_action_executed", **record})
        return record

    def _perform(self, step: str) -> None:
        bounds = self.bounds
        if step == "cancel_task":
            self.commander.cancel_task()
        elif step == "zero_cmd_vel":
            self.commander.zero_cmd_vel()
        elif step == "backup":
            self.commander.backup(bounds.backup_distance_m, bounds.backup_speed_mps)
        elif step == "spin":
            self.commander.spin(bounds.spin_angle_rad)
        elif step == "wait":
            self.commander.wait(bounds.wait_seconds)
        elif step == "clear_costmaps":
            self.commander.clear_costmaps()
        elif step == "relocalise":
            self.commander.relocalise()
        elif step == "resume_navigation":
            self.commander.resume_navigation()
        elif step == "request_assistance":
            self.commander.request_assistance()
        else:
            raise ValueError(f"unknown execution step: {step}")

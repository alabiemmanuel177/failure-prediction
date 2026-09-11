"""Publish auditable recovery decisions; live execution stays locked behind frozen evidence.

Live execution (UNTESTED LIVE) is reached only when the ``live_execution`` parameter is
true and ``configs/recovery_live_evidence.yaml`` verifies all five required evidence
items; see ``recovery_manager.live_execution``. Every guard rejection and every executed
action is republished on ``/research2/events`` as label-only diagnostics and recorded
in the per-episode sidecar ``logs/recovery/<run_id>.json``.

Hand-off from the online failure monitor (``/research2/recovery_requests``) is two
phase when the live gate is open: (1) safe stop = cancel the Nav2 task and zero
``cmd_vel``; (2) measure ``stopped`` from ``/odom`` (deployable, never denylisted)
and only then run the policy against the frozen guard with the measured state. The
guard never sees an asserted stop. Policies: R1, R2, R3 and the pilot's forced
``RP_<action>`` (``src.recovery.manager.decide_recovery``). The repeated-recovery
budget uses this node's own executed-action count, which cannot be lower than the
monitor's alarm count.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import yaml


PROJECT_ROOT = Path(os.environ.get("RESEARCH2_ROOT", Path(__file__).resolve().parents[4]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from src.recovery import GuardConfig, RecoveryRequest, RobotState, decide_recovery
from src.recovery.plumbing import MANAGER_SIDECAR_DIR, validate_policy_id
from src.recovery.selector_training import load_selector_model, predict_costs_for_request
from recovery_manager.live_execution import (
    ActionBounds, LIVE_EVIDENCE_DEFAULT, LiveExecutor, evaluate_live_gate,
)

EVENT_TOPIC = "/research2/events"
REQUEST_TOPIC = "/research2/recovery_requests"
EVENT_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=50, reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def _values(status: DiagnosticStatus) -> dict[str, str]:
    return {item.key: item.value for item in status.values}


def _boolean(value: str) -> bool:
    if value.lower() not in {"true", "false"}:
        raise ValueError(f"expected true or false, got {value!r}")
    return value.lower() == "true"


def _optional_float(value: str) -> float | None:
    return None if value.lower() in {"", "none", "null", "nan"} else float(value)


def request_from_values(values: dict[str, str]) -> RecoveryRequest:
    """Parse one monitor request; raises KeyError/ValueError on malformed input."""
    return RecoveryRequest(
        run_id=values["run_id"], warning_id=values["warning_id"],
        risk_score=float(values["risk_score"]),
        diagnosed_signal_group=values.get("diagnosed_signal_group", "unknown"),
        state=RobotState(
            stopped=_boolean(values["stopped"]),
            stop_allowed=_boolean(values["stop_allowed"]),
            localisation_poor=_boolean(values["localisation_poor"]),
            planning_stale_or_blocked=_boolean(values["planning_stale_or_blocked"]),
            rear_clearance_m=_optional_float(values["rear_clearance_m"]),
            rotation_clearance_m=_optional_float(values["rotation_clearance_m"]),
            immediate_collision_risk=_boolean(values["immediate_collision_risk"]),
            obstruction_may_be_transient=_boolean(values["obstruction_may_be_transient"]),
            relocalisation_available=_boolean(values.get("relocalisation_available", "false")),
            repeated_recovery_count=int(values["repeated_recovery_count"]),
        ),
    )


class RecoveryManager(Node):
    def __init__(self) -> None:
        super().__init__("research2_recovery_manager")
        default_config = PROJECT_ROOT / "configs/recovery_guards.yaml"
        self.declare_parameter("policy_id", "R2")
        self.declare_parameter("run_id", "UNSET")
        self.declare_parameter("guard_config", str(default_config))
        self.declare_parameter("execution_enabled", False)
        self.declare_parameter("live_execution", False)
        # Empty => the researcher-signed file; anything else is an engineering override
        # that the sidecar and the start-up log line record explicitly.
        self.declare_parameter("live_evidence", "")
        self.declare_parameter("selector_model", "")
        self.declare_parameter("goal_x", 0.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_yaw", 0.0)
        self.declare_parameter("stop_settle_seconds", 2.0)
        self.declare_parameter("stopped_linear_mps", 0.01)
        self.declare_parameter("stopped_angular_rads", 0.05)
        self.declare_parameter("output_root", "")
        if bool(self.get_parameter("execution_enabled").value):
            raise RuntimeError(
                "execution_enabled is a retired lock; live execution requires live_execution:=true "
                "and the frozen evidence file configs/recovery_live_evidence.yaml"
            )
        guard_path = Path(str(self.get_parameter("guard_config").value))
        document = yaml.safe_load(guard_path.read_text(encoding="utf-8"))
        self.guard_config = GuardConfig(
            minimum_rear_clearance_m=float(document["minimum_rear_clearance_m"]),
            minimum_rotation_clearance_m=float(document["minimum_rotation_clearance_m"]),
            maximum_repeated_recoveries=int(document["maximum_repeated_recoveries"]),
        )
        self.policy_id = validate_policy_id(str(self.get_parameter("policy_id").value))
        if self.policy_id == "R0":
            raise RuntimeError("R0 never launches the recovery manager")
        self.run_id = str(self.get_parameter("run_id").value)
        output_root = Path(str(self.get_parameter("output_root").value) or (PROJECT_ROOT / MANAGER_SIDECAR_DIR))
        self.partial_path = output_root / f"{self.run_id}.partial.json"
        self.final_path = output_root / f"{self.run_id}.json"
        self.seen_warning_ids: set[str] = set()
        self.decisions: list[dict] = []
        self.events: list[dict] = []
        self.lock = threading.Lock()
        self.publisher = self.create_publisher(DiagnosticArray, "/research2/recovery_decisions", 10)
        self.event_publisher = self.create_publisher(DiagnosticArray, EVENT_TOPIC, EVENT_QOS)
        selector_path = str(self.get_parameter("selector_model").value)
        self.selector_model = load_selector_model(Path(selector_path)) if selector_path else None
        if self.policy_id == "R3" and self.selector_model is None:
            raise RuntimeError("R3 requires selector_model (the fitted cost-sensitive selector)")
        signed_evidence = (PROJECT_ROOT / LIVE_EVIDENCE_DEFAULT).resolve()
        self.evidence_path = Path(
            str(self.get_parameter("live_evidence").value) or signed_evidence
        ).resolve()
        self.evidence_override = self.evidence_path != signed_evidence
        self.gate = evaluate_live_gate(
            bool(self.get_parameter("live_execution").value), self.evidence_path, guard_path,
        )
        if bool(self.get_parameter("live_execution").value) and not self.gate.enabled:
            raise RuntimeError(f"live execution refused at startup: {self.gate.reason}")
        self.stop_settle_seconds = float(self.get_parameter("stop_settle_seconds").value)
        self.stopped_linear = float(self.get_parameter("stopped_linear_mps").value)
        self.stopped_angular = float(self.get_parameter("stopped_angular_rads").value)
        self.measured_stopped: bool | None = None
        self.measured_at: float | None = None
        self.live_executor = None
        if self.gate.enabled:
            from recovery_manager.nav2_commander import Nav2LiveCommander  # lazy Nav2 import
            commander = Nav2LiveCommander(self)
            commander.set_mission_goal(self._goal_pose())
            self.live_executor = LiveExecutor(commander, self.gate, self._publish_event, ActionBounds())
        else:
            self.live_executor = LiveExecutor(_NullCommander(), self.gate, self._publish_event)
        override_note = (
            f" (ENGINEERING EVIDENCE OVERRIDE: {self.evidence_path})" if self.evidence_override else ""
        )
        self.get_logger().info(
            f"recovery manager policy={self.policy_id} live gate: {self.gate.reason}{override_note}"
        )
        # /odom is deployable; it runs in its own reentrant group so the stop can be
        # measured while a request callback waits for the robot to settle.
        self.odom_group = ReentrantCallbackGroup()
        self.create_subscription(
            Odometry, "/odom", self._receive_odom, qos_profile_sensor_data,
            callback_group=self.odom_group,
        )
        self.subscription = self.create_subscription(
            DiagnosticArray, REQUEST_TOPIC, self._receive, 10,
        )
        self.write_snapshot()

    # ------------------------------------------------------------------ helpers
    def _goal_pose(self):
        from geometry_msgs.msg import PoseStamped  # lazy
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.pose.position.x = float(self.get_parameter("goal_x").value)
        goal.pose.position.y = float(self.get_parameter("goal_y").value)
        yaw = float(self.get_parameter("goal_yaw").value)
        goal.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.orientation.w = math.cos(yaw / 2)
        return goal

    def _receive_odom(self, message: Odometry) -> None:
        linear = abs(float(message.twist.twist.linear.x))
        angular = abs(float(message.twist.twist.angular.z))
        self.measured_stopped = linear < self.stopped_linear and angular < self.stopped_angular
        self.measured_at = self.get_clock().now().nanoseconds * 1e-9

    def _wait_for_stop(self) -> tuple[bool, float]:
        """Poll the /odom measurement for up to ``stop_settle_seconds`` (wall time)."""
        deadline = time.monotonic() + self.stop_settle_seconds
        started = time.monotonic()
        while time.monotonic() < deadline:
            if self.measured_stopped:
                return True, time.monotonic() - started
            time.sleep(0.05)
        return bool(self.measured_stopped), time.monotonic() - started

    def _publish(self, level: int, message: str, values: dict[str, object]) -> None:
        output = DiagnosticArray()
        output.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.level = level
        status.name = "research2/recovery_decision"
        status.hardware_id = "simulation"
        status.message = message
        status.values = [
            KeyValue(key=str(key), value=(
                json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
            )) for key, value in values.items()
        ]
        output.status = [status]
        self.publisher.publish(output)

    def _publish_event(self, event: dict[str, object]) -> None:
        """Label-only /research2/events record for guard rejections and executed actions."""
        stamp = self.get_clock().now()
        output = DiagnosticArray()
        output.header.stamp = stamp.to_msg()
        output.header.frame_id = "research2_label_only"
        status = DiagnosticStatus()
        status.level = DiagnosticStatus.OK
        status.name = f"research2/{event['event_type']}"
        status.hardware_id = "simulation"
        status.message = str(event["event_type"])
        payload = {key: value for key, value in event.items() if key != "event_type"}
        status.values = [
            KeyValue(key="event_type", value=str(event["event_type"])),
            KeyValue(key="run_id", value=str(event.get("run_id", self.run_id))),
            KeyValue(key="reason", value=str(event.get("reason", event["event_type"]))),
            KeyValue(key="parameters_json", value=json.dumps(payload, sort_keys=True, default=str)),
            KeyValue(key="causal_role", value="label_only"),
        ]
        output.status = [status]
        self.event_publisher.publish(output)
        with self.lock:
            self.events.append({"time": stamp.nanoseconds * 1e-9, **json.loads(json.dumps(event, default=str))})
        # Snapshot after every event: a live sequence can outlast the episode's teardown.
        self.write_snapshot()

    # ------------------------------------------------------------------ requests
    def _receive(self, message: DiagnosticArray) -> None:
        for status in message.status:
            if status.name != "research2/recovery_request":
                continue
            values = _values(status)
            warning_id = values.get("warning_id", "")
            if warning_id in self.seen_warning_ids:
                self._publish(DiagnosticStatus.WARN, "duplicate_warning_ignored", {
                    "warning_id": warning_id, "execution_performed": False,
                })
                continue
            try:
                request = request_from_values(values)
            except (KeyError, TypeError, ValueError) as error:
                self._publish(DiagnosticStatus.ERROR, "invalid_recovery_request", {
                    "warning_id": warning_id, "error": str(error), "execution_performed": False,
                })
                continue
            self.seen_warning_ids.add(warning_id)
            record = self.handle_request(request, values)
            with self.lock:
                self.decisions.append(record)
            self.write_snapshot()

    def handle_request(self, request: RecoveryRequest, values: dict[str, str]) -> dict:
        """Two-phase hand-off; returns the sidecar record of this warning."""
        safe_stop = None
        state = request.state
        measured_stop = None
        if self.gate.enabled:
            # Always: the monitor's ``stopped`` is an estimate from its last window, and
            # the guard must only ever see the /odom measurement taken after the stop.
            safe_stop = self.live_executor.safe_stop(request.run_id, request.warning_id, self.policy_id)
            if safe_stop["execution_performed"]:
                stopped, settle = self._wait_for_stop()
                measured_stop = {"stopped": stopped, "settle_seconds": settle,
                                 "asserted_by_monitor": bool(request.state.stopped)}
                state = replace(state, stopped=bool(stopped))
        budget_count = max(state.repeated_recovery_count, self.live_executor.executed_count)
        state = replace(state, repeated_recovery_count=budget_count)
        request = replace(request, state=state)
        try:
            costs = None
            if self.policy_id == "R3":
                costs = (
                    predict_costs_for_request(
                        self.selector_model, request.risk_score,
                        request.diagnosed_signal_group, request.state,
                    ) if self.selector_model is not None
                    else json.loads(values["predicted_costs_json"])
                )
            decision = decide_recovery(
                request, self.guard_config, self.policy_id, predicted_costs=costs,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._publish(DiagnosticStatus.ERROR, "invalid_recovery_request", {
                "warning_id": request.warning_id, "error": str(error), "execution_performed": False,
            })
            return {
                "warning_id": request.warning_id, "error": str(error), "safe_stop": safe_stop,
                "recommended_action": "none", "guard_results": {}, "execution": None,
            }
        execution = self.live_executor.execute(decision)
        decision["execution_performed"] = execution["execution_performed"]
        decision["execution_status"] = execution["status"]
        decision["predicted_costs"] = costs
        self._publish(
            DiagnosticStatus.OK,
            "executed" if execution["execution_performed"] else "recommendation_only",
            decision,
        )
        return {
            **decision,
            "decision_time": _optional_float(values.get("decision_time", "none")),
            "decision_index": values.get("decision_index"),
            "request_state": {**request.state.__dict__},
            "safe_stop": safe_stop,
            "measured_stop": measured_stop,
            "execution": execution,
            "engineering_smoke": values.get("engineering_smoke", "false") == "true",
        }

    # ------------------------------------------------------------------ sidecar
    def payload(self) -> dict:
        with self.lock:
            return {
                "schema_version": 1,
                "run_id": self.run_id,
                "policy_id": self.policy_id,
                "guard_config": {
                    "minimum_rear_clearance_m": self.guard_config.minimum_rear_clearance_m,
                    "minimum_rotation_clearance_m": self.guard_config.minimum_rotation_clearance_m,
                    "maximum_repeated_recoveries": self.guard_config.maximum_repeated_recoveries,
                },
                "gate": {
                    "enabled": self.gate.enabled, "reason": self.gate.reason,
                    "evidence_path": str(self.evidence_path),
                    "evidence_override": bool(self.evidence_override),
                },
                "selector_model": str(self.get_parameter("selector_model").value) or None,
                "executed_count": self.live_executor.executed_count if self.live_executor else 0,
                "decisions": list(self.decisions),
                "events": list(self.events),
                "causal_role": "label_only",
            }

    def write_snapshot(self) -> None:
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.partial_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.payload(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8",
        )
        temporary.replace(self.partial_path)

    def write_sidecar(self) -> Path:
        if self.final_path.exists():
            raise FileExistsError(f"refusing to overwrite recovery sidecar {self.final_path}")
        self.write_snapshot()
        self.partial_path.replace(self.final_path)
        return self.final_path


class _NullCommander:
    """Never reached: the locked gate refuses before any step runs."""

    def __getattr__(self, name: str):
        def refuse(*_args, **_kwargs):
            raise RuntimeError("live execution is locked")
        return refuse


def main() -> None:
    rclpy.init()
    node = RecoveryManager()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            path = node.write_sidecar()
            node.get_logger().info(f"wrote {path}")
        except (KeyboardInterrupt, FileExistsError) as error:
            node.get_logger().error(str(error))
        finally:
            executor.shutdown()
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
            if rclpy.ok():
                rclpy.shutdown()

"""ROS 2 node: frozen predictor at 2 Hz on deployable topics, warnings to the recovery manager.

UNTESTED LIVE: the buffer, window, scoring and policy logic are exercised only through
``tests/test_failure_monitor*.py`` (replay of recorded telemetry, no ROS). Message I/O
lives here and nowhere else. The node refuses to start unless
``configs/model_freeze.yaml`` is frozen or ``smoke_unfrozen:=true`` marks every
published event ``engineering_smoke: true``.

Topics
  subscribe  /cmd_vel /odom /amcl_pose /scan /plan /local_plan
             /research2/features/perception /behavior_tree_log (mission start only)
  publish    /research2/warning            every decision (diagnostic_msgs)
             /research2/events             one ``predictor_alarm`` entry per alarm
             /research2/recovery_requests  one guarded request per alarm
Nothing on ``configs/leakage_denylist.yaml`` is ever subscribed; ``check_subscriptions``
asserts that at start-up.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(os.environ.get("RESEARCH2_ROOT", Path(__file__).resolve().parents[4]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav2_msgs.msg import BehaviorTreeLog
from nav_msgs.msg import Odometry, Path as PathMessage
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from failure_monitor.monitor_core import (
    EVENT_TOPIC, MISSION_START_TOPIC, MonitorCore, MonitorRefused, REQUEST_TOPIC, Scorer,
    WARNING_TOPIC, resolve_monitor_assets, write_sidecar,
)
from src.recovery.plumbing import MONITOR_SIDECAR_DIR


EVENT_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=50, reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
MESSAGE_TYPES = {
    "/cmd_vel": (Twist, 10),
    "/odom": (Odometry, qos_profile_sensor_data),
    "/amcl_pose": (PoseWithCovarianceStamped, 10),
    "/scan": (LaserScan, qos_profile_sensor_data),
    "/plan": (PathMessage, 10),
    "/local_plan": (PathMessage, 10),
    "/research2/features/perception": (DiagnosticArray, qos_profile_sensor_data),
}


def _kv(values: dict) -> list[KeyValue]:
    return [
        KeyValue(key=str(key), value=(
            json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, bool))
            else str(value)
        )) for key, value in values.items()
    ]


class FailureMonitor(Node):
    def __init__(self) -> None:
        super().__init__("research2_failure_monitor")
        defaults = {
            "research2_root": str(PROJECT_ROOT),
            "run_id": "UNSET",
            "recovery_policy": "R2",
            "smoke_unfrozen": False,
            "model_dir": "",
            "calibrator": "",
            "smoke_threshold": 0.5,
            "goal_x": 0.0,
            "goal_y": 0.0,
            "relocalisation_available": False,
            "output_root": "",
            "torch_threads": 1,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        values = {name: self.get_parameter(name).value for name in defaults}
        root = Path(str(values["research2_root"]))
        self.run_id = str(values["run_id"])
        output_root = Path(str(values["output_root"]) or (root / MONITOR_SIDECAR_DIR))
        self.partial_path = output_root / f"{self.run_id}.partial.json"
        self.final_path = output_root / f"{self.run_id}.json"
        try:
            assets = resolve_monitor_assets(
                root, freeze_path=root / "configs/model_freeze.yaml",
                alarm_policy_path=root / "configs/alarm_policy.yaml",
                smoke_unfrozen=bool(values["smoke_unfrozen"]),
                model_dir_override=str(values["model_dir"]),
                calibrator_override=str(values["calibrator"]),
                smoke_threshold=float(values["smoke_threshold"]) if bool(values["smoke_unfrozen"]) else None,
            )
            self.core = MonitorCore(
                run_id=self.run_id, policy_id=str(values["recovery_policy"]), assets=assets,
                root=root, goal_x=float(values["goal_x"]), goal_y=float(values["goal_y"]),
                relocalisation_available=bool(values["relocalisation_available"]),
            )
            self.core.scorer = Scorer(assets, self.core.feature_groups,
                                      threads=int(values["torch_threads"]))
        except MonitorRefused as error:
            self.get_logger().fatal(f"failure monitor refused to start: {error}")
            raise
        self.warning_pub = self.create_publisher(DiagnosticArray, WARNING_TOPIC, 10)
        self.event_pub = self.create_publisher(DiagnosticArray, EVENT_TOPIC, EVENT_QOS)
        self.request_pub = self.create_publisher(DiagnosticArray, REQUEST_TOPIC, 10)
        for topic, (message_type, qos) in MESSAGE_TYPES.items():
            self.create_subscription(
                message_type, topic, lambda message, name=topic: self.receive(name, message), qos,
            )
        self.create_subscription(BehaviorTreeLog, MISSION_START_TOPIC, self.receive_mission_start, 10)
        self.create_timer(self.core.windows.stride, self.tick)
        self.publish_event("monitor_started", {
            "policy_id": self.core.policy_id, **assets.record(),
            "subscribed_topics": list(self.core.sidecar()["subscribed_topics"]),
        })
        self.get_logger().info(
            f"failure monitor ready: policy={self.core.policy_id} threshold="
            f"{assets.alarm_policy.threshold} engineering_smoke={assets.engineering_smoke}"
        )
        self.write_snapshot()

    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def receive(self, topic: str, message) -> None:
        timestamp = self.now_seconds()
        self.core.receive(topic, message, timestamp)
        if topic == "/scan":
            self.core.receive_scan_clearances(
                message.ranges, float(message.angle_min), float(message.angle_increment), timestamp,
            )

    def receive_mission_start(self, _message: BehaviorTreeLog) -> None:
        if self.core.mission_started(self.now_seconds()):
            self.get_logger().info(f"mission start observed at {self.core.mission_start:.3f}")

    def tick(self) -> None:
        for output in self.core.step(self.now_seconds()):
            self.publish_warning(output)
            if output.alarm:
                self.publish_event("predictor_alarm", {
                    "warning_id": output.warning_id,
                    "decision_index": output.decision_index,
                    "decision_time": output.decision_time,
                    "raw_score": output.raw_score, "risk_score": output.risk_score,
                    "threshold": self.core.assets.alarm_policy.threshold,
                    "diagnosed_signal_group": output.diagnosed_signal_group,
                    "policy_id": self.core.policy_id,
                })
                self.publish_request(output)
                self.get_logger().warning(
                    f"ALARM {output.warning_id} t={output.decision_time:.3f} "
                    f"risk={output.risk_score:.3f} group={output.diagnosed_signal_group}"
                )
        if self.core.decisions and len(self.core.decisions) % 20 == 0:
            self.write_snapshot()

    def publish_warning(self, output) -> None:
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "research2_monitor"
        status = DiagnosticStatus()
        status.level = DiagnosticStatus.WARN if output.alarm else DiagnosticStatus.OK
        status.name = "research2/warning"
        status.hardware_id = "simulation"
        status.message = "alarm" if output.alarm else ("persistent" if output.persistent else "ok")
        status.values = _kv({
            "decision_index": output.decision_index, "decision_time": output.decision_time,
            "raw_score": output.raw_score, "risk_score": output.risk_score,
            "threshold": self.core.assets.alarm_policy.threshold,
            "persistent": output.persistent, "alarm": output.alarm,
            "diagnosed_signal_group": output.diagnosed_signal_group,
            "engineering_smoke": self.core.assets.engineering_smoke,
        })
        message.status = [status]
        self.warning_pub.publish(message)

    def publish_event(self, event_type: str, values: dict) -> None:
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "research2_label_only"
        status = DiagnosticStatus()
        status.level = DiagnosticStatus.OK
        status.name = f"research2/{event_type}"
        status.hardware_id = "simulation"
        status.message = event_type
        status.values = _kv({
            "event_type": event_type, "run_id": self.run_id, "reason": event_type,
            "parameters_json": json.dumps(values, sort_keys=True, separators=(",", ":")),
            "engineering_smoke": self.core.assets.engineering_smoke,
            "causal_role": "label_only",
        })
        message.status = [status]
        self.event_pub.publish(message)

    def publish_request(self, output) -> None:
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.level = DiagnosticStatus.WARN
        status.name = "research2/recovery_request"
        status.hardware_id = "simulation"
        status.message = output.warning_id
        status.values = [KeyValue(key=key, value=value)
                         for key, value in self.core.request_values(output).items()]
        message.status = [status]
        self.request_pub.publish(message)

    def write_snapshot(self) -> None:
        write_sidecar(self.core.sidecar(), self.partial_path)

    def write_sidecar(self) -> Path:
        if self.final_path.exists():
            raise FileExistsError(f"refusing to overwrite monitor sidecar {self.final_path}")
        self.write_snapshot()
        self.partial_path.replace(self.final_path)
        return self.final_path


def main() -> None:
    rclpy.init()
    node = FailureMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            path = node.write_sidecar()
            node.get_logger().info(f"wrote {path}")
        except (KeyboardInterrupt, FileExistsError) as error:
            node.get_logger().error(str(error))
        finally:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()

"""Spawn deterministic blockage or oscillation geometry at a causal onset."""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import Twist
from nav2_msgs.msg import BehaviorTreeLog
from rclpy.node import Node

from .events import EVENT_TOPIC, event_message
from .parameters import load_fault_parameters
from .schedule import FaultSchedule
from .signal_proxy import EVENT_QOS, stamp_seconds


def box_sdf(name: str, width: float, depth: float) -> str:
    return f"""<sdf version='1.9'><model name='{name}'><static>true</static><link name='link'>
      <collision name='collision'><geometry><box><size>{width} {depth} 0.8</size></box></geometry></collision>
      <visual name='visual'><geometry><box><size>{width} {depth} 0.8</size></box></geometry>
      <material><ambient>0.85 0.15 0.10 1</ambient><diffuse>0.85 0.15 0.10 1</diffuse></material>
      </visual></link></model></sdf>"""


class EnvironmentFault(Node):
    def __init__(self) -> None:
        super().__init__("research2_environment_fault")
        defaults = {
            "research2_root": os.environ.get("RESEARCH2_ROOT", str(Path.cwd())),
            "run_id": "UNSET",
            "family": "dynamic_blockage",
            "severity": "low",
            "seed": 0,
            "source_commit": "unknown",
            "world": "default",
            "injection_x": 0.0,
            "injection_y": 0.0,
            "injection_yaw": 0.0,
            "clean_prefix_seconds": 10.0,
            "planned_onset_seconds": 15.0,
            "maximum_duration_seconds": 20.0,
            "maximum_wait_seconds": 10.0,
            "minimum_command_speed_mps": 0.05,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        self.values = {name: self.get_parameter(name).value for name in defaults}
        self.family = str(self.values["family"])
        if self.family not in {"dynamic_blockage", "planner_oscillation"}:
            raise ValueError("environment_fault supports only dynamic_blockage or planner_oscillation")
        self.params = load_fault_parameters(
            Path(str(self.values["research2_root"])), self.family, str(self.values["severity"])
        )
        self.schedule = FaultSchedule(
            clean_prefix_seconds=float(self.values["clean_prefix_seconds"]),
            planned_onset_seconds=float(self.values["planned_onset_seconds"]),
            maximum_duration_seconds=min(
                float(self.values["maximum_duration_seconds"]),
                float(self.params.get("duration_seconds", self.values["maximum_duration_seconds"])),
            ),
            maximum_wait_seconds=float(self.values["maximum_wait_seconds"]),
        )
        self.last_command_speed = 0.0
        self.entities: list[str] = []
        self.removed = False
        self.event_pub = self.create_publisher(DiagnosticArray, EVENT_TOPIC, EVENT_QOS)
        self.create_subscription(Twist, "/cmd_vel", self.on_command, 10)
        self.create_subscription(BehaviorTreeLog, "/behavior_tree_log", self.on_bt, 10)
        self.create_timer(0.1, self.tick)

    def publish_event(self, event_type: str, *, eligible: bool = True, reason: str = "") -> None:
        self.event_pub.publish(event_message(
            stamp=self.get_clock().now().to_msg(), event_type=event_type,
            run_id=str(self.values["run_id"]), family=self.family,
            severity=str(self.values["severity"]), seed=int(self.values["seed"]),
            eligible=eligible, reason=reason, parameters=self.params,
            source_commit=str(self.values["source_commit"]),
        ))

    def on_command(self, message: Twist) -> None:
        self.last_command_speed = abs(float(message.linear.x)) + abs(float(message.angular.z))

    def on_bt(self, message: BehaviorTreeLog) -> None:
        # BehaviorTreeLog carries a wall timestamp on the pinned Jazzy stack; use the
        # node's simulation clock for the schedule and retain BT only as the arm signal.
        if self.schedule.arm(self.get_clock().now().nanoseconds * 1e-9):
            self.publish_event("injection_planned")

    def _spawn(self, name: str, sdf: str, x: float, y: float, yaw: float) -> bool:
        command = [
            "ros2", "run", "ros_gz_sim", "create",
            "-world", str(self.values["world"]), "-name", name,
            "-string", sdf, "-x", str(x), "-y", str(y), "-Y", str(yaw),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        if result.returncode:
            self.get_logger().error(result.stderr[-1000:])
            return False
        self.entities.append(name)
        return True

    def activate(self) -> bool:
        x = float(self.values["injection_x"])
        y = float(self.values["injection_y"])
        yaw = float(self.values["injection_yaw"])
        token = f"{int(self.values['seed']):08d}"
        if self.family == "dynamic_blockage":
            name = f"r2_blockage_{token}"
            return self._spawn(
                name,
                box_sdf(name, float(self.params["width_m"]), float(self.params["depth_m"])),
                x, y, yaw,
            )

        separation = float(self.params["pair_separation_m"])
        depth = float(self.params["obstacle_depth_m"])
        lateral_x, lateral_y = -math.sin(yaw), math.cos(yaw)
        successes = []
        for index, direction in enumerate((-1.0, 1.0)):
            name = f"r2_oscillation_{token}_{index}"
            successes.append(self._spawn(
                name, box_sdf(name, 0.35, depth),
                x + direction * lateral_x * separation / 2,
                y + direction * lateral_y * separation / 2,
                yaw,
            ))
        return all(successes)

    def remove(self) -> None:
        for name in self.entities:
            result = subprocess.run(
                ["ros2", "run", "ros_gz_sim", "remove", "-world",
                 str(self.values["world"]), "-name", name],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode:
                self.get_logger().error(f"failed to remove {name}: {result.stderr[-1000:]}")
        self.removed = True

    def tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        state = self.schedule.update(
            now,
            eligible=self.last_command_speed >= float(self.values["minimum_command_speed_mps"]),
        )
        if state == "activated":
            if self.activate():
                self.publish_event("injection_started")
            else:
                self.schedule.rejection_time = now
                self.publish_event("injection_error", eligible=False, reason="entity spawn failed")
        elif state == "rejected":
            self.publish_event("injection_ineligible", eligible=False,
                               reason="eligibility deadline expired")
        elif state == "complete" and self.entities and not self.removed:
            self.remove()
            self.publish_event("injection_ended")


def main() -> None:
    rclpy.init()
    node = EnvironmentFault()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

"""Proxy raw simulator streams and activate one deterministic signal fault."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav2_msgs.msg import BehaviorTreeLog
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan

from .events import EVENT_TOPIC, event_message
from .parameters import load_fault_parameters
from .schedule import FaultSchedule
from .transforms import biased_progress, camera_occlusion, lidar_dropout, semantic_risk_corruption


EVENT_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class SignalProxy(Node):
    def __init__(self) -> None:
        super().__init__("research2_signal_proxy")
        defaults = {
            "research2_root": os.environ.get("RESEARCH2_ROOT", str(Path.cwd())),
            "run_id": "UNSET",
            "family": "none",
            "severity": "none",
            "seed": 0,
            "source_commit": "unknown",
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
        self.passive = self.family in {"none", "dynamic_blockage", "planner_oscillation"}
        self.severity = str(self.values["severity"])
        self.seed = int(self.values["seed"])
        self.params = load_fault_parameters(
            Path(str(self.values["research2_root"])), self.family, self.severity
        )
        self.schedule = FaultSchedule(
            clean_prefix_seconds=float(self.values["clean_prefix_seconds"]),
            planned_onset_seconds=float(self.values["planned_onset_seconds"]),
            maximum_duration_seconds=float(self.values["maximum_duration_seconds"]),
            maximum_wait_seconds=float(self.values["maximum_wait_seconds"]),
        )
        self.last_command_speed = 0.0
        self.last_state = "waiting_for_goal"
        self.previous_odom_input: tuple[float, float] | None = None
        self.previous_odom_output: tuple[float, float] | None = None
        self.slip_yaw_bias = 0.0
        self.last_odom_time: float | None = None
        self.localisation_applied = False

        self.event_pub = self.create_publisher(DiagnosticArray, EVENT_TOPIC, EVENT_QOS)
        self.image_pub = self.create_publisher(Image, "/camera/image", qos_profile_sensor_data)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self.odom_pub = self.create_publisher(Odometry, "/odom", qos_profile_sensor_data)
        self.semantic_pub = self.create_publisher(
            OccupancyGrid, "/semantic/risk_grid", qos_profile_sensor_data
        )
        self.semantic_odom_pub = self.create_publisher(
            OccupancyGrid, "/semantic/risk_grid_odom", qos_profile_sensor_data
        )
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)

        self.create_subscription(Image, "/research2/raw/camera/image", self.on_image, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/research2/raw/scan", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/research2/raw/odom", self.on_odom, qos_profile_sensor_data)
        self.create_subscription(Twist, "/cmd_vel", self.on_command, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self.on_amcl, 10)
        self.create_subscription(BehaviorTreeLog, "/behavior_tree_log", self.on_bt, 10)
        self.create_subscription(DiagnosticArray, EVENT_TOPIC, self.on_event, EVENT_QOS)
        self.create_subscription(
            OccupancyGrid, "/research2/raw/semantic/risk_grid", self.on_semantic,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            OccupancyGrid, "/research2/raw/semantic/risk_grid_odom", self.on_semantic_odom,
            qos_profile_sensor_data,
        )

    def publish_event(self, event_type: str, stamp, *, eligible: bool = True, reason: str = "") -> None:
        self.event_pub.publish(event_message(
            stamp=stamp,
            event_type=event_type,
            run_id=str(self.values["run_id"]),
            family=self.family,
            severity=self.severity,
            seed=self.seed,
            eligible=eligible,
            reason=reason,
            parameters=self.params,
            source_commit=str(self.values["source_commit"]),
        ))

    def on_bt(self, message: BehaviorTreeLog) -> None:
        if self.passive:
            return
        # Nav2's BehaviorTreeLog timestamp remains on wall time after the G6
        # wall-to-simulation-time activation boundary. It is only a goal-start signal;
        # schedule all causal timing on this node's simulation clock.
        now = self.get_clock().now().to_msg()
        if self.schedule.arm(stamp_seconds(now)):
            self.publish_event("injection_planned", now)

    def on_event(self, message: DiagnosticArray) -> None:
        """Arm from the controller's retained goal marker; BT logging is a fallback.

        One validation episode completed navigation without the transient BT callback
        reaching this node.  The controller publishes goal_dispatched reliably after
        recording starts, so it is the primary causal arm signal and also remains in
        the bag.  Fault events published by this node are ignored by event type.
        """
        if self.passive:
            return
        for status in message.status:
            values = {item.key: item.value for item in status.values}
            if values.get("event_type") != "goal_dispatched":
                continue
            if values.get("run_id") != str(self.values["run_id"]):
                continue
            stamp = message.header.stamp
            if self.schedule.arm(stamp_seconds(stamp)):
                self.publish_event("injection_planned", stamp)

    def on_command(self, message: Twist) -> None:
        self.last_command_speed = abs(float(message.linear.x)) + abs(float(message.angular.z))

    def state(self, source_family: str, stamp, *, source_eligible: bool = True) -> str:
        if self.family != source_family:
            return "inactive"
        now = stamp_seconds(stamp)
        motion_required = self.family in {
            "lidar_dropout", "wheel_slip", "localisation_perturbation",
            "dynamic_blockage", "planner_oscillation",
        }
        motion_ok = self.last_command_speed >= float(self.values["minimum_command_speed_mps"])
        result = self.schedule.update(now, eligible=source_eligible and (motion_ok or not motion_required))
        if result == "activated":
            self.publish_event("injection_started", stamp)
        elif result == "rejected":
            self.publish_event("injection_ineligible", stamp, eligible=False,
                               reason="eligibility deadline expired")
        elif result == "complete" and self.last_state == "active":
            self.publish_event("injection_ended", stamp)
        self.last_state = "active" if result == "activated" else result
        return self.last_state

    def on_image(self, message: Image) -> None:
        state = self.state(
            "camera_occlusion", message.header.stamp,
            source_eligible=message.height > 0 and message.width > 0,
        )
        if self.family != "camera_occlusion" or state != "active":
            self.image_pub.publish(message)
            return
        if message.encoding not in {"rgb8", "bgr8"} or message.step != message.width * 3:
            self.publish_event("injection_error", message.header.stamp, eligible=False,
                               reason=f"unsupported image encoding or stride: {message.encoding}")
            return
        array = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.width, 3)
        output = camera_occlusion(
            array,
            mask_fraction=float(self.params["mask_fraction"]),
            dropout_probability=float(self.params["dropout_probability"]),
            seed=self.seed,
            stamp_ns=stamp_ns(message.header.stamp),
        )
        if output is None:
            return
        shifted = copy.deepcopy(message)
        shifted.data = output.tobytes()
        self.image_pub.publish(shifted)

    def on_scan(self, message: LaserScan) -> None:
        state = self.state(
            "lidar_dropout", message.header.stamp, source_eligible=bool(message.ranges)
        )
        output = copy.deepcopy(message)
        if self.family == "lidar_dropout" and state == "active":
            output.ranges = lidar_dropout(
                list(message.ranges), invalid_fraction=float(self.params["invalid_fraction"]),
                seed=self.seed,
            )
        self.scan_pub.publish(output)

    def on_odom(self, message: Odometry) -> None:
        state = self.state("wheel_slip", message.header.stamp)
        output = copy.deepcopy(message)
        current = (float(message.pose.pose.position.x), float(message.pose.pose.position.y))
        if self.previous_odom_input is None:
            self.previous_odom_input = current
            self.previous_odom_output = current
        elif self.family == "wheel_slip" and state == "active":
            assert self.previous_odom_output is not None
            adjusted = biased_progress(
                self.previous_odom_input, current, self.previous_odom_output,
                float(self.params["odometry_progress_scale"]),
            )
            output.pose.pose.position.x, output.pose.pose.position.y = adjusted
            output.twist.twist.linear.x *= float(self.params["odometry_progress_scale"])
            now = stamp_seconds(message.header.stamp)
            dt = 0.0 if self.last_odom_time is None else max(0.0, now - self.last_odom_time)
            self.slip_yaw_bias += float(self.params["yaw_bias_rate_rad_s"]) * dt
            q = output.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            yaw += self.slip_yaw_bias
            q.x = q.y = 0.0
            q.z, q.w = math.sin(yaw / 2), math.cos(yaw / 2)
            self.previous_odom_output = adjusted
            self.last_odom_time = now
        else:
            self.previous_odom_output = current
            self.slip_yaw_bias = 0.0
            self.last_odom_time = stamp_seconds(message.header.stamp)
        self.previous_odom_input = current
        self.odom_pub.publish(output)

    def on_amcl(self, message: PoseWithCovarianceStamped) -> None:
        state = self.state("localisation_perturbation", message.header.stamp)
        if self.family != "localisation_perturbation" or state != "active" or self.localisation_applied:
            return
        output = copy.deepcopy(message)
        yaw_offset = float(self.params["yaw_rad"])
        translation = float(self.params["translation_m"])
        direction = -1.0 if self.seed % 2 else 1.0
        output.pose.pose.position.x += direction * translation
        q = output.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        yaw += direction * yaw_offset
        q.x = q.y = 0.0
        q.z, q.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.initial_pose_pub.publish(output)
        self.localisation_applied = True

    def corrupted_semantic(self, message: OccupancyGrid) -> OccupancyGrid:
        state = self.state(
            "semantic_corruption", message.header.stamp, source_eligible=bool(message.data)
        )
        output = copy.deepcopy(message)
        if self.family == "semantic_corruption" and state == "active":
            output.data = semantic_risk_corruption(
                list(message.data),
                corruption_probability=float(self.params["corruption_probability"]),
                risk_scale=float(self.params["risk_scale"]),
                seed=self.seed,
                stamp_ns=stamp_ns(message.header.stamp),
            )
        return output

    def on_semantic(self, message: OccupancyGrid) -> None:
        output = self.corrupted_semantic(message)
        self.semantic_pub.publish(output)

    def on_semantic_odom(self, message: OccupancyGrid) -> None:
        output = self.corrupted_semantic(message)
        self.semantic_odom_pub.publish(output)


def main() -> None:
    rclpy.init()
    node = SignalProxy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

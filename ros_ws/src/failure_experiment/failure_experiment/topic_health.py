"""Independent message counters for post-recording MCAP reconciliation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan

from .events import EVENT_TOPIC


class TopicHealth(Node):
    def __init__(self) -> None:
        super().__init__("research2_topic_health")
        self.declare_parameter("run_id", "UNSET")
        self.declare_parameter(
            "output_root",
            os.path.join(
                os.environ.get("RESEARCH2_ROOT", str(Path.cwd())), "logs/topic-health"
            ),
        )
        self.run_id = str(self.get_parameter("run_id").value)
        self.output_root = Path(str(self.get_parameter("output_root").value))
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.partial_path = self.output_root / f"{self.run_id}.partial.json"
        self.final_path = self.output_root / f"{self.run_id}.json"
        self.counts: dict[str, int] = {}
        self.last_receive_ns: dict[str, int] = {}
        self.maximum_gap_seconds: dict[str, float] = {}
        self.recording_window_started = False
        self.recording_window_ended = False
        self.recording_window_active = False
        self.publisher = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        subscriptions = (
            ("/scan", LaserScan, qos_profile_sensor_data),
            ("/odom", Odometry, qos_profile_sensor_data),
            ("/camera/image", Image, qos_profile_sensor_data),
            ("/cmd_vel", Twist, 10),
            ("/amcl_pose", PoseWithCovarianceStamped, 10),
            (EVENT_TOPIC, DiagnosticArray, 10),
        )
        for topic, message_type, qos in subscriptions:
            self.counts[topic] = 0
            self.maximum_gap_seconds[topic] = 0.0
            callback = (
                self.receive_event
                if topic == EVENT_TOPIC
                else lambda _message, name=topic: self.receive(name)
            )
            self.create_subscription(message_type, topic, callback, qos)
        self.create_timer(1.0, self.publish_health)

    def receive(self, topic: str) -> None:
        if self.recording_window_ended:
            return
        now = self.get_clock().now().nanoseconds
        previous = self.last_receive_ns.get(topic)
        if previous is not None:
            self.maximum_gap_seconds[topic] = max(
                self.maximum_gap_seconds[topic], (now - previous) * 1e-9
            )
        self.last_receive_ns[topic] = now
        self.counts[topic] += 1

    def receive_event(self, message: DiagnosticArray) -> None:
        event_type = None
        for status in message.status:
            values = {item.key: item.value for item in status.values}
            if "event_type" in values:
                event_type = values["event_type"]
                break
        if event_type == "recording_window_started":
            self.counts = {topic: 0 for topic in self.counts}
            self.last_receive_ns.clear()
            self.maximum_gap_seconds = {
                topic: 0.0 for topic in self.maximum_gap_seconds
            }
            self.recording_window_started = True
            self.recording_window_active = True
        self.receive(EVENT_TOPIC)
        if event_type == "recording_window_ended":
            self.recording_window_ended = True
            self.recording_window_active = False
            self.write_snapshot()

    def publish_health(self) -> None:
        now = self.get_clock().now()
        statuses = []
        for topic, count in sorted(self.counts.items()):
            # Event counts are required for integrity reconciliation but are label-only.
            # Publishing them on deployable /diagnostics would reveal injection activity.
            if topic == EVENT_TOPIC:
                continue
            age = None
            if topic in self.last_receive_ns:
                age = max(0.0, (now.nanoseconds - self.last_receive_ns[topic]) * 1e-9)
            stale = count == 0 or (age is not None and age > 2.0)
            status = DiagnosticStatus()
            status.level = DiagnosticStatus.STALE if stale else DiagnosticStatus.OK
            status.name = f"research2/topic_health{topic}"
            status.message = "stale" if stale else "receiving"
            status.hardware_id = "simulation"
            status.values = [
                KeyValue(key="run_id", value=self.run_id),
                KeyValue(key="topic", value=topic),
                KeyValue(key="received_count", value=str(count)),
                KeyValue(key="age_seconds", value="unknown" if age is None else f"{age:.6f}"),
                KeyValue(
                    key="maximum_gap_seconds",
                    value=f"{self.maximum_gap_seconds[topic]:.6f}",
                ),
            ]
            statuses.append(status)
        message = DiagnosticArray()
        message.header.stamp = now.to_msg()
        message.header.frame_id = "research2_monitor"
        message.status = statuses
        self.publisher.publish(message)
        self.write_snapshot()

    def payload(self) -> dict:
        return {
            "run_id": self.run_id,
            "counts": self.counts,
            "maximum_gap_seconds": self.maximum_gap_seconds,
            "recording_window_started": self.recording_window_started,
            "recording_window_ended": self.recording_window_ended,
            "clock": "simulation_time",
        }

    def write_snapshot(self) -> None:
        temporary = self.partial_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(self.partial_path)

    def write_sidecar(self) -> Path:
        if self.final_path.exists():
            raise FileExistsError(
                f"refusing to overwrite topic-health sidecar {self.final_path}"
            )
        self.write_snapshot()
        self.partial_path.replace(self.final_path)
        return self.final_path


def main() -> None:
    rclpy.init()
    node = TopicHealth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            path = node.write_sidecar()
            node.get_logger().info(f"wrote {path}")
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

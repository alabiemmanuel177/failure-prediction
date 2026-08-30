"""Publish a single auditable episode, terminal, recovery, or protocol event."""

from __future__ import annotations

import json

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node

from .events import EVENT_TOPIC, event_message
from .signal_proxy import EVENT_QOS


class EventMarker(Node):
    def __init__(self) -> None:
        super().__init__("research2_event_marker")
        for name, default in {
            "event_type": "manual_marker",
            "run_id": "UNSET",
            "family": "none",
            "severity": "none",
            "seed": 0,
            "eligible": True,
            "reason": "",
            "parameters_json": "{}",
            "source_commit": "unknown",
        }.items():
            self.declare_parameter(name, default)
        self.publisher = self.create_publisher(DiagnosticArray, EVENT_TOPIC, EVENT_QOS)
        self.timer = self.create_timer(0.2, self.publish_once)

    def publish_once(self) -> None:
        self.publisher.publish(event_message(
            stamp=self.get_clock().now().to_msg(),
            event_type=str(self.get_parameter("event_type").value),
            run_id=str(self.get_parameter("run_id").value),
            family=str(self.get_parameter("family").value),
            severity=str(self.get_parameter("severity").value),
            seed=int(self.get_parameter("seed").value),
            eligible=bool(self.get_parameter("eligible").value),
            reason=str(self.get_parameter("reason").value),
            parameters=json.loads(str(self.get_parameter("parameters_json").value)),
            source_commit=str(self.get_parameter("source_commit").value),
        ))
        self.get_logger().info("event published")
        self.timer.cancel()
        self.create_timer(0.5, rclpy.shutdown)


def main() -> None:
    rclpy.init()
    node = EventMarker()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == "__main__":
    main()


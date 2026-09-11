"""Publish synchronized scalar summaries instead of recording semantic images."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from .semantic_summary_core import summarize_semantic_arrays


TOPIC = "/research2/features/perception"
KINDS = {
    "/semantic/probabilities": "probabilities",
    "/semantic/classes": "classes",
    "/semantic/confidence": "confidence",
    "/semantic/uncertainty": "uncertainty",
    "/semantic/inference_latency_ms": "latency_ms",
}


def stamp_ns(message: Image) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def image_array(message: Image) -> np.ndarray:
    encoding = message.encoding.lower()
    if encoding in {"mono8", "8uc1"}:
        dtype, channels = np.uint8, 1
    elif encoding.startswith("32fc"):
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
        suffix = encoding.removeprefix("32fc")
        channels = int(suffix or 1)
    else:
        raise ValueError(f"unsupported semantic encoding: {message.encoding}")
    item_size = np.dtype(dtype).itemsize
    row_values = int(message.step) // item_size
    raw = np.frombuffer(message.data, dtype=dtype)
    rows = raw.reshape(int(message.height), row_values)
    values = rows[:, : int(message.width) * channels]
    if channels == 1:
        return values.reshape(int(message.height), int(message.width)).copy()
    return values.reshape(int(message.height), int(message.width), channels).copy()


class SemanticSummary(Node):
    def __init__(self) -> None:
        super().__init__("research2_semantic_summary")
        self.publisher = self.create_publisher(DiagnosticArray, TOPIC, qos_profile_sensor_data)
        self.buffers: dict[str, dict[int, tuple[Image, np.ndarray]]] = defaultdict(dict)
        self.previous_classes: np.ndarray | None = None
        self.published_frames = 0
        for topic, kind in KINDS.items():
            self.create_subscription(
                Image, topic, lambda message, name=kind: self.receive(name, message),
                qos_profile_sensor_data,
            )

    def receive(self, kind: str, message: Image) -> None:
        stamp = stamp_ns(message)
        try:
            values = image_array(message)
        except ValueError as error:
            self.get_logger().error(str(error))
            return
        self.buffers[kind][stamp] = (message, values)
        for buffer in self.buffers.values():
            while len(buffer) > 20:
                del buffer[min(buffer)]
        if not all(stamp in self.buffers[name] for name in KINDS.values()):
            return
        messages = {name: self.buffers[name].pop(stamp) for name in KINDS.values()}
        arrays = {name: item[1] for name, item in messages.items()}
        try:
            summary = summarize_semantic_arrays(
                probabilities=arrays["probabilities"],
                classes=arrays["classes"],
                confidence=arrays["confidence"],
                uncertainty=arrays["uncertainty"],
                latency_ms=arrays["latency_ms"],
                previous_classes=self.previous_classes,
            )
        except ValueError as error:
            self.get_logger().error(f"semantic frame {stamp} rejected: {error}")
            return
        self.previous_classes = arrays["classes"]
        self.published_frames += 1
        source = messages["classes"][0]
        status = DiagnosticStatus()
        status.level = DiagnosticStatus.OK
        status.name = "research2/perception_summary"
        status.message = "synchronized causal semantic scalars"
        status.hardware_id = "simulation_perception"
        status.values = [
            KeyValue(key=name, value=format(value, ".9g"))
            for name, value in sorted(summary.items())
        ]
        status.values.append(KeyValue(
            key="published_frame_count", value=str(self.published_frames)
        ))
        output = DiagnosticArray()
        output.header = source.header
        output.status = [status]
        self.publisher.publish(output)


def main() -> None:
    rclpy.init()
    node = SemanticSummary()
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

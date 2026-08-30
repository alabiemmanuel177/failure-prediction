"""Pure scalar summaries for recorded ROS telemetry messages."""

from __future__ import annotations

import math
import struct
from typing import Sequence


def scan_summaries(ranges: Sequence[float], angle_min: float, angle_increment: float) -> dict[str, float]:
    total = len(ranges)
    if total == 0:
        raise ValueError("scan has no beams")
    finite = [(index, float(value)) for index, value in enumerate(ranges)
              if math.isfinite(float(value)) and float(value) > 0]
    valid_fraction = len(finite) / total

    def sector_min(low: float, high: float) -> float:
        values = [value for index, value in finite
                  if low <= angle_min + index * angle_increment < high]
        return min(values) if values else 0.0

    return {
        "valid_return_fraction": valid_fraction,
        "minimum_front_range": sector_min(-math.pi / 6, math.pi / 6),
        "minimum_left_range": sector_min(math.pi / 6, 5 * math.pi / 6),
        "minimum_right_range": sector_min(-5 * math.pi / 6, -math.pi / 6),
    }


def covariance_trace(covariance: Sequence[float]) -> float:
    if len(covariance) != 36:
        raise ValueError("pose covariance must have 36 elements")
    return sum(float(covariance[index]) for index in (0, 7, 35))


def path_summaries(points: Sequence[tuple[float, float]]) -> dict[str, float]:
    if not points:
        return {"path_length": 0.0, "path_curvature": 0.0}
    length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    turns = []
    for a, b, c in zip(points, points[1:], points[2:]):
        first = math.atan2(b[1] - a[1], b[0] - a[0])
        second = math.atan2(c[1] - b[1], c[0] - b[0])
        difference = math.atan2(math.sin(second - first), math.cos(second - first))
        segment = math.dist(b, c)
        if segment > 0:
            turns.append(abs(difference) / segment)
    return {"path_length": length, "path_curvature": sum(turns) / len(turns) if turns else 0.0}


def float_image_mean(message) -> float:
    if str(message.encoding).lower() != "32fc1":
        raise ValueError(f"expected 32FC1 image, got {message.encoding!r}")
    endian = ">" if bool(message.is_bigendian) else "<"
    width, height = int(message.width), int(message.height)
    row_bytes = int(message.step)
    values = []
    data = bytes(message.data)
    for row in range(height):
        start = row * row_bytes
        values.extend(struct.unpack_from(f"{endian}{width}f", data, start))
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        raise ValueError("float image has no finite values")
    return sum(finite) / len(finite)


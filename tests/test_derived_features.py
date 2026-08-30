import math
import struct
from types import SimpleNamespace

import pytest

from src.features.derived import covariance_trace, float_image_mean, path_summaries, scan_summaries


def test_scan_summaries_preserve_dropout_and_sector_health():
    ranges = [float("inf")] * 12
    ranges[5] = 2.0; ranges[6] = 1.0; ranges[9] = 3.0
    result = scan_summaries(ranges, -math.pi, math.pi / 6)
    assert result["valid_return_fraction"] == 0.25
    assert result["minimum_front_range"] == 1.0


def test_covariance_trace_uses_planar_pose_terms():
    covariance = [0.0] * 36
    covariance[0], covariance[7], covariance[35] = 1.0, 2.0, 3.0
    assert covariance_trace(covariance) == 6.0
    with pytest.raises(ValueError):
        covariance_trace([0.0])


def test_path_summaries_measure_length_and_turning():
    straight = path_summaries([(0, 0), (1, 0), (2, 0)])
    turning = path_summaries([(0, 0), (1, 0), (1, 1)])
    assert straight == {"path_length": 2.0, "path_curvature": 0.0}
    assert turning["path_curvature"] == pytest.approx(math.pi / 2)


def test_float_image_mean_honours_stride_and_endianness():
    row = struct.pack("<2f", 1.0, 3.0) + b"PAD!"
    message = SimpleNamespace(
        encoding="32FC1", is_bigendian=False, width=2, height=1, step=len(row), data=row,
    )
    assert float_image_mean(message) == 2.0

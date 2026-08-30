"""Pure deterministic fault transformations, independent of ROS execution."""

from __future__ import annotations

import math

import numpy as np


def frame_seed(seed: int, stamp_ns: int) -> int:
    return (int(seed) * 1_000_003 + int(stamp_ns)) % (2**63 - 1)


def camera_occlusion(image: np.ndarray, *, mask_fraction: float,
                     dropout_probability: float, seed: int, stamp_ns: int) -> np.ndarray | None:
    if not 0.0 <= mask_fraction <= 1.0 or not 0.0 <= dropout_probability <= 1.0:
        raise ValueError("camera probabilities must lie in [0, 1]")
    rng = np.random.default_rng(frame_seed(seed, stamp_ns))
    if dropout_probability and rng.random() < dropout_probability:
        return None
    output = image.copy()
    width = round(output.shape[1] * mask_fraction)
    if width:
        output[:, output.shape[1] - width :] = 0
    return output


def lidar_dropout(ranges: list[float], *, invalid_fraction: float,
                  seed: int) -> list[float]:
    if not 0.0 <= invalid_fraction <= 1.0:
        raise ValueError("invalid_fraction must lie in [0, 1]")
    output = list(ranges)
    count = round(len(output) * invalid_fraction)
    if not output or not count:
        return output
    start = (int(seed) * 104_729) % len(output)
    for offset in range(count):
        output[(start + offset) % len(output)] = math.inf
    return output


def semantic_risk_corruption(values: list[int], *, corruption_probability: float,
                             risk_scale: float, seed: int, stamp_ns: int) -> list[int]:
    if not 0.0 <= corruption_probability <= 1.0:
        raise ValueError("corruption_probability must lie in [0, 1]")
    if not 0.0 <= risk_scale <= 1.0:
        raise ValueError("risk_scale must lie in [0, 1]")
    output = np.asarray(values, dtype=np.int16).copy()
    valid = output >= 0
    rng = np.random.default_rng(frame_seed(seed, stamp_ns))
    mask = valid & (rng.random(output.shape) < corruption_probability)
    output[mask] = np.rint(output[mask] * risk_scale).astype(np.int16)
    return output.tolist()


def biased_progress(previous_input: tuple[float, float], current_input: tuple[float, float],
                    previous_output: tuple[float, float], scale: float) -> tuple[float, float]:
    if not 0.0 <= scale <= 1.0:
        raise ValueError("progress scale must lie in [0, 1]")
    dx = current_input[0] - previous_input[0]
    dy = current_input[1] - previous_input[1]
    return previous_output[0] + scale * dx, previous_output[1] + scale * dy


"""Pure array reduction for causal semantic telemetry summaries."""

from __future__ import annotations

import math

import numpy as np


def summarize_semantic_arrays(
    *,
    probabilities: np.ndarray,
    classes: np.ndarray,
    confidence: np.ndarray,
    uncertainty: np.ndarray,
    latency_ms: np.ndarray,
    previous_classes: np.ndarray | None,
    class_count: int = 6,
    ignore_index: int = 255,
) -> dict[str, float]:
    """Reduce one synchronized semantic frame without retaining image payloads."""

    probabilities = np.asarray(probabilities, dtype=np.float32)
    classes = np.asarray(classes)
    confidence = np.asarray(confidence, dtype=np.float32)
    uncertainty = np.asarray(uncertainty, dtype=np.float32)
    latency_ms = np.asarray(latency_ms, dtype=np.float32)
    if probabilities.ndim != 3 or probabilities.shape[-1] != class_count:
        raise ValueError(f"probabilities must have shape HxWx{class_count}")
    spatial = probabilities.shape[:2]
    for name, value in (
        ("classes", classes), ("confidence", confidence),
        ("uncertainty", uncertainty),
    ):
        if value.shape != spatial:
            raise ValueError(f"{name} shape {value.shape} differs from {spatial}")
    if not all(np.isfinite(value).all() for value in
               (probabilities, confidence, uncertainty, latency_ms)):
        raise ValueError("semantic arrays must contain only finite values")
    clipped = np.clip(probabilities, np.finfo(np.float32).tiny, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=-1)
    result = {
        "confidence_mean": float(np.mean(confidence)),
        "uncertainty_mean": float(np.mean(uncertainty)),
        "entropy_mean": float(np.mean(entropy)),
        "inference_latency_ms": float(np.mean(latency_ms)),
    }
    valid = classes != ignore_index
    valid_count = int(np.count_nonzero(valid))
    for class_id in range(class_count):
        result[f"class_{class_id}_fraction"] = (
            float(np.count_nonzero((classes == class_id) & valid) / valid_count)
            if valid_count else 0.0
        )
    result["ignored_pixel_fraction"] = float(1.0 - valid_count / classes.size)
    if previous_classes is not None and previous_classes.shape == classes.shape:
        comparable = valid & (previous_classes != ignore_index)
        count = int(np.count_nonzero(comparable))
        if count:
            result["temporal_disagreement"] = float(
                np.count_nonzero(classes[comparable] != previous_classes[comparable]) / count
            )
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("semantic summary produced a non-finite value")
    return result

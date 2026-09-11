from pathlib import Path
import sys

import numpy as np
import pytest
from types import SimpleNamespace


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "ros_ws/src/failure_experiment"
sys.path.insert(0, str(PACKAGE_ROOT))

from failure_experiment.semantic_summary_core import summarize_semantic_arrays
from scripts.extract_bag_scalar_telemetry import feature_values


def test_semantic_summary_is_finite_and_preserves_class_fractions():
    probabilities = np.array([
        [[0.8, 0.2, 0, 0, 0, 0], [0.1, 0.9, 0, 0, 0, 0]],
        [[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]],
    ], dtype=np.float32)
    classes = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    result = summarize_semantic_arrays(
        probabilities=probabilities,
        classes=classes,
        confidence=probabilities.max(axis=-1),
        uncertainty=1.0 - probabilities.max(axis=-1),
        latency_ms=np.array([[8.0]], dtype=np.float32),
        previous_classes=np.array([[0, 0], [2, 255]], dtype=np.uint8),
    )
    assert result["confidence_mean"] == pytest.approx(0.925)
    assert result["inference_latency_ms"] == 8.0
    assert result["class_0_fraction"] == pytest.approx(1 / 3)
    assert result["class_1_fraction"] == pytest.approx(1 / 3)
    assert result["class_2_fraction"] == pytest.approx(1 / 3)
    assert result["ignored_pixel_fraction"] == 0.25
    assert result["temporal_disagreement"] == pytest.approx(1 / 3)


def test_first_frame_omits_temporal_disagreement():
    probabilities = np.full((2, 2, 6), 1 / 6, dtype=np.float32)
    result = summarize_semantic_arrays(
        probabilities=probabilities,
        classes=np.zeros((2, 2), dtype=np.uint8),
        confidence=np.full((2, 2), 1 / 6, dtype=np.float32),
        uncertainty=np.ones((2, 2), dtype=np.float32),
        latency_ms=np.array([5.0], dtype=np.float32),
        previous_classes=None,
    )
    assert "temporal_disagreement" not in result
    assert result["entropy_mean"] == pytest.approx(np.log(6), rel=1e-6)


def test_semantic_summary_rejects_shape_and_nonfinite_evidence():
    probabilities = np.full((2, 2, 6), 1 / 6, dtype=np.float32)
    with pytest.raises(ValueError, match="classes shape"):
        summarize_semantic_arrays(
            probabilities=probabilities,
            classes=np.zeros((1, 2), dtype=np.uint8),
            confidence=np.ones((2, 2), dtype=np.float32),
            uncertainty=np.ones((2, 2), dtype=np.float32),
            latency_ms=np.array([1.0], dtype=np.float32),
            previous_classes=None,
        )
    bad = probabilities.copy(); bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        summarize_semantic_arrays(
            probabilities=bad,
            classes=np.zeros((2, 2), dtype=np.uint8),
            confidence=np.ones((2, 2), dtype=np.float32),
            uncertainty=np.ones((2, 2), dtype=np.float32),
            latency_ms=np.array([1.0], dtype=np.float32),
            previous_classes=None,
        )


def test_scalar_bag_adapter_exports_only_cross_profile_primary_fields():
    values = [
        SimpleNamespace(key="confidence_mean", value="0.8"),
        SimpleNamespace(key="uncertainty_mean", value="0.2"),
        SimpleNamespace(key="inference_latency_ms", value="8.5"),
        SimpleNamespace(key="temporal_disagreement", value="0.1"),
        SimpleNamespace(key="class_0_fraction", value="0.7"),
    ]
    message = SimpleNamespace(status=[SimpleNamespace(
        name="research2/perception_summary", values=values
    )])
    assert feature_values("/research2/features/perception", message) == {
        "confidence_mean": 0.8,
        "uncertainty_mean": 0.2,
        "inference_latency_ms": 8.5,
    }

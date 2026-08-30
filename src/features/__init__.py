"""Past-only feature extraction and leakage controls."""

from .causal import FeatureSpec, ScalarSample, extract_decision_rows
from .leakage import LeakageError, LeakagePolicy
from .normalization import NormalizationBundle
from .window_features import derive_window_features

__all__ = [
    "FeatureSpec", "ScalarSample", "extract_decision_rows",
    "LeakageError", "LeakagePolicy", "NormalizationBundle", "derive_window_features",
]

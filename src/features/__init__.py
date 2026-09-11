"""Past-only feature extraction and leakage controls."""

from .causal import FeatureSpec, ScalarSample, extract_decision_rows
from .artifacts import validate_decision_artifact, validate_sequence_artifact
from .leakage import LeakageError, LeakagePolicy
from .normalization import NormalizationBundle
from .schema import load_primary_feature_set, load_raw_feature_contract, resolve_feature_specs
from .sequences import SequenceExample, assemble_causal_sequences, model_columns
from .window_features import derive_window_features

__all__ = [
    "FeatureSpec", "ScalarSample", "extract_decision_rows",
    "LeakageError", "LeakagePolicy", "NormalizationBundle", "derive_window_features",
    "load_raw_feature_contract", "resolve_feature_specs",
    "load_primary_feature_set",
    "SequenceExample", "assemble_causal_sequences", "model_columns",
    "validate_decision_artifact", "validate_sequence_artifact",
]

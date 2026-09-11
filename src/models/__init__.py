"""Transparent and learned predictor implementations.

Torch-backed architectures live in ``tcn`` (P3), ``recurrent`` (P4), ``transformer``
(P5) and ``autoencoder`` (P2) and are imported lazily through
``src.models.common.build_model`` so this package stays importable without torch.
``oracle`` (P6) is a label-only analysis upper bound.
"""

from .threshold_rules import Rule, ThresholdRuleSet

__all__ = ["Rule", "ThresholdRuleSet"]

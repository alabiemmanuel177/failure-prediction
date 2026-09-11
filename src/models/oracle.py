"""P6: oracle warning (ANALYSIS-ONLY upper bound; never an operational method).

The oracle reads the label-only ``eligibility`` array and returns risk 1.0 on every
``eligible_positive`` decision and 0.0 otherwise. It has no parameters, sees no
features, and exists only to bound what a perfect predictor could achieve under the
frozen alarm policy. It must never appear in a deployable configuration or a
freeze record as a predictor.
"""

from __future__ import annotations

import numpy as np

from .common import ELIGIBLE_POSITIVE, ORACLE_MODEL_ID

ANALYSIS_ONLY = True
MODEL_ID = ORACLE_MODEL_ID


def oracle_risk_scores(eligibility: np.ndarray) -> np.ndarray:
    values = np.asarray([str(value) for value in np.asarray(eligibility).tolist()])
    return (values == ELIGIBLE_POSITIVE).astype(np.float64)


__all__ = ["ANALYSIS_ONLY", "MODEL_ID", "oracle_risk_scores"]

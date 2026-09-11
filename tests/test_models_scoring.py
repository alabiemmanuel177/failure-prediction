import numpy as np
import pytest

from src.models.autoencoder import ErrorScaler
from src.models.oracle import ANALYSIS_ONLY, oracle_risk_scores


def test_oracle_scores_one_on_eligible_positive_and_zero_elsewhere_and_is_analysis_only():
    eligibility = np.asarray(["eligible_negative", "eligible_positive", "excluded_too_late",
                              "excluded_near_event_or_injection", "eligible_positive"])
    assert oracle_risk_scores(eligibility).tolist() == [0.0, 1.0, 0.0, 0.0, 1.0]
    assert ANALYSIS_ONLY is True


def test_error_scaler_is_monotone_bounded_and_development_only():
    rng = np.random.default_rng(0)
    errors = rng.gamma(2.0, 1.0, 1000)
    scaler = ErrorScaler.fit(errors, lower_quantile=0.05, upper_quantile=0.95)
    grid = np.linspace(-1.0, 30.0, 500)
    scores = scaler.transform(grid)
    assert (np.diff(scores) >= 0).all()
    assert scores.min() == 0.0 and scores.max() == 1.0
    assert (scaler.transform(errors) <= 1.0).all() and (scaler.transform(errors) >= 0.0).all()
    assert ErrorScaler.from_dict(scaler.to_dict()) == scaler
    with pytest.raises(ValueError):
        ErrorScaler.fit(errors, split="validation")
    constant = ErrorScaler.fit(np.ones(10))
    assert constant.upper > constant.lower


def test_autoencoder_returns_per_window_reconstruction_error():
    torch = pytest.importorskip("torch")
    from src.models.autoencoder import ReconstructionAutoencoder
    torch.manual_seed(0)
    model = ReconstructionAutoencoder(840, hidden_sizes=[64], bottleneck=8, dropout=0.0).eval()
    x = torch.randn(3, 10, 84)
    with torch.no_grad():
        errors = model(x)
    assert errors.shape == (3,) and (errors >= 0).all()

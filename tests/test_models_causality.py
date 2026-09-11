import importlib.util
from pathlib import Path

import pytest

from src.models.common import build_model, count_parameters, load_model_config

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("p3_causal_tcn", "p4_gru", "p5_compact_transformer")


def _model(model_id):
    import torch
    torch.manual_seed(0)
    config = load_model_config(ROOT / "configs/models" / f"{model_id}.yaml", model_id)
    return build_model(model_id, config).eval(), config


@pytest.mark.parametrize("model_id", MODELS)
def test_perturbing_a_future_step_never_changes_an_earlier_output(model_id):
    import torch
    model, _ = _model(model_id)
    torch.manual_seed(1)
    x = torch.randn(3, 10, 84)
    with torch.no_grad():
        base = model.forward_sequence(x)
        for t in range(9):
            perturbed = x.clone()
            perturbed[:, t + 1:, :] += torch.randn_like(perturbed[:, t + 1:, :]) * 5.0
            out = model.forward_sequence(perturbed)
            assert torch.allclose(out[:, :t + 1], base[:, :t + 1], atol=1e-5), (model_id, t)
            assert not torch.allclose(out[:, t + 1], base[:, t + 1]), (model_id, t)


@pytest.mark.parametrize("model_id", MODELS)
def test_forward_returns_the_last_step_logit(model_id):
    import torch
    model, _ = _model(model_id)
    x = torch.randn(4, 10, 84)
    with torch.no_grad():
        assert torch.allclose(model(x), model.forward_sequence(x)[:, -1])
        assert model(x).shape == (4,)


def test_transformer_mask_is_strictly_upper_triangular_minus_inf():
    import torch
    from src.models.transformer import CompactCausalTransformer
    mask = CompactCausalTransformer.causal_mask(4, torch.device("cpu"))
    assert torch.isinf(mask).sum() == 6
    assert (mask.tril() == 0).all()


def test_parameter_budget_p4_within_25_percent_of_p3_and_p5_recorded():
    counts = {model_id: count_parameters(_model(model_id)[0]) for model_id in MODELS}
    p3 = counts["p3_causal_tcn"]
    assert 0.75 * p3 <= counts["p4_gru"] <= 1.25 * p3, counts
    assert 0.5 * p3 <= counts["p5_compact_transformer"] <= 1.5 * p3, counts
    assert 50_000 <= p3 <= 150_000, counts

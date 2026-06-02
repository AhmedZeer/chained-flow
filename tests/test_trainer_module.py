import torch

from chained_flow.drafters.hidden_mlp import HiddenMLPConfig
from chained_flow.training.losses import DrafterLossConfig
from chained_flow.training.trainer_module import HiddenMLPTrainingModule


def test_trainer_module_forward_returns_custom_loss(fake_wrapper):
    module = HiddenMLPTrainingModule(
        fake_wrapper,
        HiddenMLPConfig(context_size=2, draft_length=2),
        DrafterLossConfig(),
    )
    context_hidden = torch.randn(3, 2, fake_wrapper.model.config.hidden_size)
    target_hidden = torch.randn(3, 2, fake_wrapper.model.config.hidden_size)
    future_tokens = torch.tensor([[1, 2], [2, 3], [3, 4]])

    output = module(
        context_hidden=context_hidden,
        target_hidden=target_hidden,
        future_tokens=future_tokens,
    )

    assert output["loss"].ndim == 0
    assert output["pred_hidden"].shape == target_hidden.shape
    assert "loss_component/hidden.mse" in output
    output["loss"].backward()
    assert any(param.grad is not None for param in module.drafter.parameters())


def test_lm_head_buffers_are_not_persistent(fake_wrapper):
    module = HiddenMLPTrainingModule(
        fake_wrapper,
        HiddenMLPConfig(context_size=2, draft_length=2),
        DrafterLossConfig(),
    )
    state_keys = set(module.state_dict().keys())
    assert "lm_head_weight" not in state_keys
    assert all(not key.startswith("lm_head") for key in state_keys)

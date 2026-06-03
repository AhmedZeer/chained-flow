import torch
import pytest

from chained_flow.training.eval_vae import logit_kl_divergence, torch_dtype_from_string


def test_logit_kl_divergence_is_zero_for_matching_logits():
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    assert logit_kl_divergence(logits, logits).item() == pytest.approx(0.0, abs=1e-6)


def test_torch_dtype_from_string_accepts_float16_alias():
    assert torch_dtype_from_string("fp16") is torch.float16

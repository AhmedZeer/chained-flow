import torch
import pytest

from chained_flow.training.eval_vae import (
    logit_kl_divergence,
    per_token_logit_kl_divergence,
    per_token_vae_metrics,
    summarize_metric,
    torch_dtype_from_string,
)


def test_logit_kl_divergence_is_zero_for_matching_logits():
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    assert logit_kl_divergence(logits, logits).item() == pytest.approx(0.0, abs=1e-6)


def test_torch_dtype_from_string_accepts_float16_alias():
    assert torch_dtype_from_string("fp16") is torch.float16


def test_per_token_logit_kl_divergence_returns_one_value_per_token():
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

    kl = per_token_logit_kl_divergence(logits, logits)

    assert kl.shape == (2,)
    assert kl.tolist() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_summarize_metric_reports_distribution_stats():
    summary = summarize_metric(torch.tensor([1.0, 2.0, 3.0]))

    assert summary["mean"] == pytest.approx(2.0)
    assert summary["std"] == pytest.approx(0.816496, rel=1e-5)
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0
    assert summary["p50"] == 2.0
    assert set(summary) == {"mean", "std", "min", "max", "p50", "p90", "p95", "p99"}


def test_per_token_vae_metrics_reports_all_eval_metrics():
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    logits = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    metrics = per_token_vae_metrics(
        hidden,
        hidden,
        mu=torch.zeros(2, 2),
        logvar=torch.zeros(2, 2),
        real_logits=logits,
        recon_logits=logits,
    )

    assert set(metrics) == {"hidden.mse", "hidden.cos", "hidden.norm", "latent.kl", "logit.kl", "token.match"}
    assert metrics["hidden.mse"].tolist() == [0.0, 0.0]
    assert metrics["token.match"].tolist() == [1.0, 1.0]

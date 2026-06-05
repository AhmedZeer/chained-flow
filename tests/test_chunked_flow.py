import json

import pytest
import torch

from chained_flow.drafters.chunked_flow import CrossAttentionFlowExpert, SingleExpertFlowConfig, SingleExpertFlowDrafter
from chained_flow.training.train_chunked_flow import FlowLossArguments, SingleExpertFlowTrainingModule
from chained_flow.vae import HiddenVAEConfig, build_hidden_vae


def write_vae_checkpoint(path, *, hidden_size=8, latent_size=3, intermediate_size=5):
    path.mkdir()
    config = {
        "model_args": {
            "vae_type": "mlp",
            "hidden_size": hidden_size,
            "latent_size": latent_size,
            "intermediate_size": intermediate_size,
            "device": None,
        },
        "loss_args": {},
        "data_args": {},
    }
    (path / "chained_flow_vae_config.json").write_text(json.dumps(config), encoding="utf-8")
    vae = build_hidden_vae(
        "mlp",
        HiddenVAEConfig(hidden_size=hidden_size, latent_size=latent_size, intermediate_size=intermediate_size),
    )
    torch.save({f"vae.{key}": value for key, value in vae.state_dict().items()}, path / "pytorch_model.bin")


def flow_config(vae_dir, *, context_size=3, num_flow_steps=1):
    return SingleExpertFlowConfig(
        context_size=context_size,
        draft_length=2,
        chunk_size=2,
        vae_dir=str(vae_dir),
        expert_dim=8,
        num_heads=2,
        ffn_multiplier=2,
        num_flow_steps=num_flow_steps,
    )


def test_cross_attention_flow_expert_returns_velocity_shape():
    expert = CrossAttentionFlowExpert(
        latent_size=3,
        chunk_size=2,
        expert_dim=8,
        num_heads=2,
        ffn_multiplier=2,
    )
    z_tau = torch.randn(4, 2, 3)
    tau = torch.rand(4)
    context = torch.randn(4, 3, 3)

    velocity = expert(z_tau, tau, context)

    assert velocity.shape == (4, 2, 3)


def test_single_expert_flow_config_rejects_non_phase_one_shapes(tmp_path):
    with pytest.raises(ValueError, match="draft_length=2"):
        SingleExpertFlowConfig(vae_dir=str(tmp_path), draft_length=3, chunk_size=3)
    with pytest.raises(ValueError, match="chunk_size == draft_length"):
        SingleExpertFlowConfig(vae_dir=str(tmp_path), draft_length=2, chunk_size=1)


def test_single_expert_flow_drafter_predicts_and_proposes(fake_wrapper, tmp_path):
    vae_dir = tmp_path / "vae"
    write_vae_checkpoint(vae_dir, hidden_size=fake_wrapper.model.config.hidden_size)
    drafter = SingleExpertFlowDrafter(fake_wrapper, flow_config(vae_dir))
    input_ids = torch.tensor([[1, 2, 3]])
    state, _ = fake_wrapper.prefill(input_ids)

    context = torch.randn(4, 3, fake_wrapper.model.config.hidden_size)
    latents = drafter.predict_latent_from_context(context)
    hidden = drafter.decode_latent(latents)
    proposal = drafter.propose(state, 2)

    assert latents.shape == (4, 2, 3)
    assert hidden.shape == (4, 2, fake_wrapper.model.config.hidden_size)
    assert proposal.tokens.shape == (1, 2)
    assert proposal.hidden_states.shape == (1, 2, fake_wrapper.model.config.hidden_size)
    assert proposal.latent_states.shape == (1, 2, 3)
    assert proposal.logits.shape == (1, 2, fake_wrapper.model.config.hidden_size)
    assert all(not parameter.requires_grad for parameter in drafter.vae.parameters())


def test_single_expert_flow_training_module_returns_finite_loss_and_gradients(fake_wrapper, tmp_path):
    vae_dir = tmp_path / "vae"
    write_vae_checkpoint(vae_dir, hidden_size=fake_wrapper.model.config.hidden_size)
    module = SingleExpertFlowTrainingModule(
        fake_wrapper,
        flow_config(vae_dir, context_size=2),
        FlowLossArguments(),
    )
    context_hidden = torch.randn(3, 2, fake_wrapper.model.config.hidden_size)
    target_hidden = torch.randn(3, 2, fake_wrapper.model.config.hidden_size)
    future_tokens = torch.tensor([[1, 2], [2, 3], [3, 4]])

    output = module(
        context_hidden=context_hidden,
        target_hidden=target_hidden,
        future_tokens=future_tokens,
    )
    output["loss"].backward()

    assert torch.isfinite(output["loss"])
    assert output["loss"].ndim == 0
    assert output["pred_latent"].shape == (3, 2, 3)
    assert output["pred_hidden"].shape == target_hidden.shape
    for name in [
        "loss_component/flow.mse",
        "loss_component/latent.mse",
        "loss_component/hidden.rel_mse",
        "loss_component/hidden.cos",
        "loss_component/logit.ce",
        "loss_component/verifier.expected_accept",
    ]:
        assert name in output
    assert any(parameter.grad is not None for parameter in module.drafter.expert.parameters())
    assert all(parameter.grad is None for parameter in module.drafter.vae.parameters())


def test_lm_head_buffers_are_not_persistent(fake_wrapper, tmp_path):
    vae_dir = tmp_path / "vae"
    write_vae_checkpoint(vae_dir, hidden_size=fake_wrapper.model.config.hidden_size)
    module = SingleExpertFlowTrainingModule(
        fake_wrapper,
        flow_config(vae_dir, context_size=2),
        FlowLossArguments(),
    )

    state_keys = set(module.state_dict().keys())

    assert "lm_head_weight" not in state_keys
    assert all(not key.startswith("lm_head") for key in state_keys)
    assert all(".vae." not in key for key in state_keys)

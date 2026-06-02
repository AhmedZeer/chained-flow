import torch

from chained_flow.drafters.hidden_mlp import HiddenMLPConfig, HiddenMLPDrafter


def test_lm_head_matches_model_logits(fake_wrapper):
    input_ids = torch.tensor([[1, 2, 3]])
    state, _ = fake_wrapper.prefill(input_ids)
    projected = fake_wrapper.lm_head(state.final_hidden)
    assert torch.equal(projected.argmax(dim=-1), state.logits.argmax(dim=-1))


def test_latest_hidden_predicts_same_next_token(fake_wrapper):
    input_ids = torch.tensor([[1, 2, 3]])
    state, _ = fake_wrapper.prefill(input_ids)
    latest_logits = fake_wrapper.lm_head(fake_wrapper.latest_hidden(state).unsqueeze(1))[:, -1, :]
    assert latest_logits.argmax(dim=-1).tolist() == state.logits[:, -1, :].argmax(dim=-1).tolist()


def test_hidden_mlp_predict_from_context_shape(fake_wrapper):
    drafter = HiddenMLPDrafter(fake_wrapper, HiddenMLPConfig(context_size=3, draft_length=2))
    context = torch.randn(4, 3, fake_wrapper.model.config.hidden_size)
    pred = drafter.predict_from_context(context)
    assert pred.shape == (4, 2, fake_wrapper.model.config.hidden_size)

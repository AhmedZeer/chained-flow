import torch


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

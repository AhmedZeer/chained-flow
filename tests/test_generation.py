import torch

from chained_flow.context import ChainedFlowContext
from chained_flow.drafters.ar import ARDrafter
from chained_flow.drafters.base import DraftResult
from chained_flow.generation import generate_with_drafter


def test_generation_with_ar_drafter_matches_shift_pattern(fake_wrapper):
    result = generate_with_drafter(
        ChainedFlowContext(fake_wrapper),
        ARDrafter(fake_wrapper),
        torch.tensor([[1, 2]]),
        max_new_tokens=4,
        draft_len=2,
        eos_token_id=None,
    )
    assert result.generated_ids.tolist() == [[1, 2, 3, 4, 5, 6]]
    assert [step.accepted_len for step in result.step_stats] == [2]


class WrongDrafter:
    def propose(self, state, max_tokens):
        return DraftResult(tokens=torch.full((1, max_tokens), 0, dtype=torch.long))


def test_generation_with_wrong_drafter_falls_back_to_verifier(fake_wrapper):
    result = generate_with_drafter(
        ChainedFlowContext(fake_wrapper),
        WrongDrafter(),
        torch.tensor([[1, 2]]),
        max_new_tokens=3,
        draft_len=2,
        eos_token_id=None,
    )
    assert result.generated_ids.tolist() == [[1, 2, 3, 4, 5]]
    assert [step.accepted_len for step in result.step_stats] == [0, 0]

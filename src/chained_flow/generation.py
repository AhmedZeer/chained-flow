from __future__ import annotations

from dataclasses import dataclass, field

import torch

from chained_flow.context import ChainedFlowContext
from chained_flow.drafters.base import BaseDrafter
from chained_flow.frozen_lm import LMState
from chained_flow.timing import TimingStats, timed_section
from chained_flow.verifier import SpeculativeVerifier


@dataclass
class GenerationStepStats:
    draft_len: int
    accepted_len: int
    generated_count: int
    timings: TimingStats = field(default_factory=TimingStats)


@dataclass
class GenerationResult:
    input_ids: torch.Tensor
    generated_ids: torch.Tensor
    step_stats: list[GenerationStepStats]
    timings: TimingStats

    @property
    def generated_token_count(self) -> int:
        return self.generated_ids.shape[1] - self.input_ids.shape[1]


def _append_state_token_ids(state: LMState, token_ids: torch.Tensor) -> LMState:
    return LMState(
        input_ids=torch.cat([state.input_ids, token_ids], dim=1),
        past_key_values=state.past_key_values,
        final_hidden=state.final_hidden,
        logits=state.logits,
        position=state.position + token_ids.shape[1],
    )


@torch.inference_mode()
def generate_with_drafter(
    context: ChainedFlowContext,
    drafter: BaseDrafter,
    prompt: str | torch.Tensor,
    *,
    max_new_tokens: int,
    draft_len: int,
    eos_token_id: int | None = None,
) -> GenerationResult:
    frozen_lm = context.frozen_lm
    timings = TimingStats()
    step_stats: list[GenerationStepStats] = []
    eos_token_id = frozen_lm.eos_token_id if eos_token_id is None else eos_token_id

    with timed_section(timings, "total_generation", frozen_lm.device):
        input_ids = frozen_lm.tokenize(prompt) if isinstance(prompt, str) else prompt.to(frozen_lm.device)
        state, prefill_timings = frozen_lm.prefill(input_ids)
        timings.merge(prefill_timings)

        verifier = SpeculativeVerifier(frozen_lm)
        generated = 0

        while generated < max_new_tokens:
            anchor_token, next_timings = frozen_lm.next_token(state)
            timings.merge(next_timings)
            state, anchor_timings = frozen_lm.forward_with_cache(anchor_token, state, use_cache=True)
            timings.merge(anchor_timings)
            generated += 1

            if eos_token_id is not None and int(anchor_token.item()) == int(eos_token_id):
                break

            remaining_after_anchor = max_new_tokens - generated
            if remaining_after_anchor <= 0 or draft_len <= 0:
                break

            proposal = drafter.propose(state, min(draft_len, remaining_after_anchor))
            timings.merge(proposal.timings, "drafter")
            verify_result = verifier.verify(state, proposal.tokens)
            timings.merge(verify_result.timings, "verifier")
            state = verify_result.state

            accepted = verify_result.acceptance.accepted_len
            emitted = accepted
            generated += emitted
            step_stats.append(
                GenerationStepStats(
                    draft_len=proposal.tokens.shape[1],
                    accepted_len=accepted,
                    generated_count=emitted,
                    timings=verify_result.timings,
                )
            )

            if eos_token_id is not None:
                new_tokens = state.input_ids[:, -emitted:]
                if emitted > 0 and (new_tokens == eos_token_id).any():
                    break

        total = timings.get("total_generation")
        if total > 0:
            timings.add("tokens_per_second", generated / total)

    return GenerationResult(
        input_ids=input_ids,
        generated_ids=state.input_ids,
        step_stats=step_stats,
        timings=timings,
    )

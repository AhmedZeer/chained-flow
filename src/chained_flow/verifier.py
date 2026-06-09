from __future__ import annotations

from dataclasses import dataclass, field
import copy

import torch

from chained_flow.acceptance import AcceptanceResult, greedy_acceptance
from chained_flow.frozen_lm import FrozenLMWrapper, LMState
from chained_flow.timing import TimingStats, timed_section


@dataclass
class VerifyResult:
    state: LMState
    acceptance: AcceptanceResult
    timings: TimingStats = field(default_factory=TimingStats)


class SpeculativeVerifier:
    def __init__(self, frozen_lm: FrozenLMWrapper):
        self.frozen_lm = frozen_lm

    @torch.inference_mode()
    def verify(
        self,
        state: LMState,
        draft_tokens: torch.Tensor,
    ) -> VerifyResult:
        timings = TimingStats()
        device = self.frozen_lm.device
        draft_tokens = draft_tokens.to(device)

        verify_state = LMState(
            input_ids=state.input_ids,
            past_key_values=copy.deepcopy(state.past_key_values),
            final_hidden=state.final_hidden,
            logits=state.logits,
            position=state.position,
        )
        with timed_section(timings, "verifier_forward", device):
            verified_state, forward_timings = self.frozen_lm.forward_with_cache(draft_tokens, verify_state, use_cache=True)
        timings.merge(forward_timings)

        with timed_section(timings, "acceptance", device):
            draft_position_logits = verified_state.logits[
                :,
                state.position : state.position + draft_tokens.shape[1],
                :,
            ]
            verifier_tokens = torch.cat(
                [
                    state.logits[:, -1:, :].argmax(dim=-1),
                    draft_position_logits.argmax(dim=-1),
                ],
                dim=1,
            )
            acceptance = greedy_acceptance(draft_tokens, verifier_tokens)

        with timed_section(timings, "cache_repair", device):
            committed_count = acceptance.accepted_len
            if committed_count == draft_tokens.shape[1]:
                repaired_state = verified_state
            elif committed_count == 0:
                repaired_state = state
            else:
                committed_tokens = draft_tokens[:, :committed_count]
                repaired_state, repair_timings = self.frozen_lm.forward_with_cache(
                    committed_tokens,
                    state,
                    use_cache=True,
                )
                timings.merge(repair_timings)

        return VerifyResult(state=repaired_state, acceptance=acceptance, timings=timings)

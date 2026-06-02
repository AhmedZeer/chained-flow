from __future__ import annotations

from dataclasses import dataclass, field

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

        with timed_section(timings, "verifier_forward", device):
            verified_state, forward_timings = self.frozen_lm.forward_with_cache(draft_tokens, state, use_cache=True)
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
            committed_end = state.position + committed_count
            if hasattr(verified_state.past_key_values, "crop"):
                verified_state.past_key_values.crop(committed_end)
            committed_ids = torch.cat(
                [state.input_ids, draft_tokens[:, :committed_count]],
                dim=1,
            )
            verified_state = LMState(
                input_ids=committed_ids,
                past_key_values=verified_state.past_key_values,
                final_hidden=verified_state.final_hidden[:, : committed_ids.shape[1], :],
                logits=verified_state.logits[:, : committed_ids.shape[1], :],
                position=committed_ids.shape[1],
            )

        return VerifyResult(state=verified_state, acceptance=acceptance, timings=timings)

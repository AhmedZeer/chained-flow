from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chained_flow.drafters.base import DraftResult
from chained_flow.frozen_lm import FrozenLMWrapper, LMState
from chained_flow.timing import TimingStats, timed_section


@dataclass
class HiddenMLPConfig:
    context_size: int = 4
    draft_length: int = 4
    hidden_multiplier: int = 2


class HiddenMLPDrafter(nn.Module):
    def __init__(
        self,
        frozen_lm: FrozenLMWrapper,
        config: HiddenMLPConfig | None = None,
    ):
        super().__init__()
        self.frozen_lm = frozen_lm
        self.config = config or HiddenMLPConfig()
        hidden_size = frozen_lm.model.config.hidden_size
        mlp_hidden = hidden_size * self.config.hidden_multiplier
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size * self.config.context_size),
            nn.Linear(hidden_size * self.config.context_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size * self.config.draft_length),
        )

    def _context(self, state: LMState) -> torch.Tensor:
        hidden = state.final_hidden
        if hidden.shape[1] >= self.config.context_size:
            return hidden[:, -self.config.context_size :, :]
        pad_len = self.config.context_size - hidden.shape[1]
        pad = hidden[:, :1, :].expand(-1, pad_len, -1)
        return torch.cat([pad, hidden], dim=1)

    def predict_hidden(self, state: LMState, max_tokens: int | None = None) -> torch.Tensor:
        draft_len = self.config.draft_length if max_tokens is None else min(max_tokens, self.config.draft_length)
        ctx = self._context(state).reshape(state.input_ids.shape[0], -1)
        return self.net(ctx).reshape(
            state.input_ids.shape[0],
            self.config.draft_length,
            -1,
        )[:, :draft_len, :]

    def forward(self, state: LMState, max_tokens: int | None = None) -> torch.Tensor:
        return self.predict_hidden(state, max_tokens=max_tokens)

    @torch.inference_mode()
    def propose(self, state: LMState, max_tokens: int) -> DraftResult:
        timings = TimingStats()
        draft_len = min(max_tokens, self.config.draft_length)
        if draft_len <= 0:
            empty = torch.empty((state.input_ids.shape[0], 0), dtype=torch.long, device=self.frozen_lm.device)
            return DraftResult(tokens=empty, timings=timings)

        with timed_section(timings, "drafter_hidden_mlp", self.frozen_lm.device):
            future_hidden = self.predict_hidden(state, max_tokens=draft_len)
            logits = self.frozen_lm.lm_head(future_hidden)
            tokens = logits.argmax(dim=-1)
        return DraftResult(tokens=tokens, hidden_states=future_hidden, logits=logits, timings=timings)

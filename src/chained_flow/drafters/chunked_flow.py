from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chained_flow.drafters.base import DraftResult
from chained_flow.frozen_lm import FrozenLMWrapper, LMState
from chained_flow.timing import TimingStats, timed_section


@dataclass
class SingleExpertFlowConfig:
    context_size: int = 4
    draft_length: int = 2
    chunk_size: int = 2
    vae_dir: str | None = None
    expert_dim: int = 128
    num_heads: int = 4
    ffn_multiplier: int = 4
    num_flow_steps: int = 1
    noise_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.draft_length < 1:
            raise ValueError("draft_length must be >= 1")
        if self.chunk_size != self.draft_length:
            raise ValueError("SingleExpertFlowDrafter requires chunk_size == draft_length")
        if self.num_flow_steps < 1:
            raise ValueError("num_flow_steps must be >= 1")
        if self.vae_dir is None:
            raise ValueError("SingleExpertFlowDrafter requires vae_dir")


class CrossAttentionFlowExpert(nn.Module):
    def __init__(
        self,
        *,
        latent_size: int,
        chunk_size: int,
        expert_dim: int,
        num_heads: int,
        ffn_multiplier: int = 4,
    ):
        super().__init__()
        if expert_dim % num_heads != 0:
            raise ValueError("expert_dim must be divisible by num_heads")
        self.latent_size = latent_size
        self.chunk_size = chunk_size
        self.query_proj = nn.Linear(latent_size, expert_dim)
        self.key_proj = nn.Linear(latent_size, expert_dim)
        self.value_proj = nn.Linear(latent_size, expert_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, expert_dim),
            nn.SiLU(),
            nn.Linear(expert_dim, expert_dim),
        )
        self.slot_embedding = nn.Embedding(chunk_size, expert_dim)
        self.query_norm = nn.LayerNorm(expert_dim)
        self.context_norm = nn.LayerNorm(expert_dim)
        self.self_attn = nn.MultiheadAttention(expert_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(expert_dim, num_heads, batch_first=True)
        ffn_dim = expert_dim * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.LayerNorm(expert_dim),
            nn.Linear(expert_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, expert_dim),
        )
        self.out_norm = nn.LayerNorm(expert_dim)
        self.out_proj = nn.Linear(expert_dim, latent_size)

    def forward(self, z_tau: torch.Tensor, tau: torch.Tensor, context_latents: torch.Tensor) -> torch.Tensor:
        if z_tau.ndim != 3:
            raise ValueError("z_tau must have shape [B, C, Z]")
        if context_latents.ndim != 3:
            raise ValueError("context_latents must have shape [B, m, Z]")
        if z_tau.shape[1] != self.chunk_size:
            raise ValueError(f"z_tau chunk length must be {self.chunk_size}, got {z_tau.shape[1]}")
        if z_tau.shape[-1] != self.latent_size or context_latents.shape[-1] != self.latent_size:
            raise ValueError("z_tau and context_latents must use the configured latent size")

        batch = z_tau.shape[0]
        if tau.ndim == 1:
            tau = tau[:, None]
        elif tau.ndim == 3:
            tau = tau.reshape(batch, 1)
        elif tau.ndim != 2:
            raise ValueError("tau must have shape [B], [B, 1], or [B, 1, 1]")
        tau = tau.to(device=z_tau.device, dtype=z_tau.dtype)

        slot_ids = torch.arange(self.chunk_size, device=z_tau.device)
        q = self.query_proj(z_tau)
        q = q + self.time_mlp(tau).unsqueeze(1)
        q = q + self.slot_embedding(slot_ids).unsqueeze(0)
        q = self.query_norm(q)

        k = self.context_norm(self.key_proj(context_latents))
        v = self.context_norm(self.value_proj(context_latents))

        self_out, _ = self.self_attn(q, q, q, need_weights=False)
        q = q + self_out
        cross_out, _ = self.cross_attn(q, k, v, need_weights=False)
        q = q + cross_out
        q = q + self.ffn(q)
        return self.out_proj(self.out_norm(q))


class SingleExpertFlowDrafter(nn.Module):
    def __init__(self, frozen_lm: FrozenLMWrapper, config: SingleExpertFlowConfig):
        super().__init__()
        self.frozen_lm = frozen_lm
        self.config = config
        from chained_flow.vae import load_hidden_vae_from_dir

        self.vae = load_hidden_vae_from_dir(config.vae_dir, device=frozen_lm.device, freeze=True)
        self.hidden_size = frozen_lm.model.config.hidden_size
        self.latent_size = self.vae.config.latent_size
        self.expert = CrossAttentionFlowExpert(
            latent_size=self.latent_size,
            chunk_size=config.chunk_size,
            expert_dim=config.expert_dim,
            num_heads=config.num_heads,
            ffn_multiplier=config.ffn_multiplier,
        )

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {key: value for key, value in state.items() if not key.startswith("vae.")}

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        full_state = super().state_dict()
        full_state.update(state_dict)
        return super().load_state_dict(full_state, strict=strict, assign=assign)

    def _context(self, state: LMState) -> torch.Tensor:
        hidden = state.final_hidden
        if hidden.shape[1] >= self.config.context_size:
            return hidden[:, -self.config.context_size :, :]
        pad_len = self.config.context_size - hidden.shape[1]
        pad = hidden[:, :1, :].expand(-1, pad_len, -1)
        return torch.cat([pad, hidden], dim=1)

    def _vae_dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    def _expert_dtype(self) -> torch.dtype:
        return next(self.expert.parameters()).dtype

    def encode_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        original_shape = hidden.shape[:-1]
        flat_hidden = hidden.reshape(-1, hidden.shape[-1]).to(device=self.frozen_lm.device, dtype=self._vae_dtype())
        with torch.no_grad():
            latent = self.vae.encode(flat_hidden).mu
        return latent.reshape(*original_shape, -1)

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        original_shape = latent.shape[:-1]
        flat_latent = latent.reshape(-1, latent.shape[-1]).to(device=self.frozen_lm.device, dtype=self._vae_dtype())
        decoded = self.vae.decode(flat_latent)
        return decoded.reshape(*original_shape, -1)

    def integrate_latents(
        self,
        context_latents: torch.Tensor,
        *,
        z0: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        if context_latents.ndim != 3:
            raise ValueError("context_latents must have shape [B, m, Z]")
        context_latents = context_latents.to(device=self.frozen_lm.device, dtype=self._expert_dtype())
        batch = context_latents.shape[0]
        steps = self.config.num_flow_steps if num_steps is None else num_steps
        if steps < 1:
            raise ValueError("num_steps must be >= 1")
        if z0 is None:
            z = torch.randn(
                batch,
                self.config.chunk_size,
                self.latent_size,
                device=context_latents.device,
                dtype=context_latents.dtype,
            ) * self.config.noise_scale
        else:
            z = z0.to(device=context_latents.device, dtype=context_latents.dtype)
        dt = 1.0 / steps
        for step in range(steps):
            tau = torch.full((batch,), step * dt, device=context_latents.device, dtype=context_latents.dtype)
            z = z + dt * self.expert(z, tau, context_latents)
        return z

    def predict_latent_from_context(
        self,
        context_hidden: torch.Tensor,
        max_tokens: int | None = None,
        *,
        z0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_hidden.ndim != 3:
            raise ValueError("context_hidden must have shape [B, m, D]")
        if context_hidden.shape[1] != self.config.context_size:
            raise ValueError(
                f"context_hidden must have context size {self.config.context_size}, got {context_hidden.shape[1]}"
            )
        draft_len = self.config.draft_length if max_tokens is None else min(max_tokens, self.config.draft_length)
        context_latents = self.encode_hidden(context_hidden)
        pred_latents = self.integrate_latents(context_latents, z0=z0)
        return pred_latents[:, :draft_len, :]

    def predict_hidden(self, state: LMState, max_tokens: int | None = None) -> torch.Tensor:
        latent = self.predict_latent_from_context(self._context(state), max_tokens=max_tokens)
        return self.decode_latent(latent)

    def forward(self, state: LMState, max_tokens: int | None = None) -> torch.Tensor:
        return self.predict_hidden(state, max_tokens=max_tokens)

    @torch.inference_mode()
    def propose(self, state: LMState, max_tokens: int) -> DraftResult:
        timings = TimingStats()
        draft_len = min(max_tokens, self.config.draft_length)
        if draft_len <= 0:
            empty = torch.empty((state.input_ids.shape[0], 0), dtype=torch.long, device=self.frozen_lm.device)
            return DraftResult(tokens=empty, timings=timings)

        with timed_section(timings, "drafter_single_expert_flow", self.frozen_lm.device):
            future_latent = self.predict_latent_from_context(self._context(state), max_tokens=draft_len)
            future_hidden = self.decode_latent(future_latent)
            logits = self.frozen_lm.lm_head(future_hidden)
            tokens = logits.argmax(dim=-1)
        return DraftResult(
            tokens=tokens,
            hidden_states=future_hidden,
            latent_states=future_latent,
            logits=logits,
            timings=timings,
        )

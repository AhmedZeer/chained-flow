from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HiddenVAEConfig:
    hidden_size: int = 1024
    latent_size: int = 256
    intermediate_size: int = 512


@dataclass
class LatentDistribution:
    mu: torch.Tensor
    logvar: torch.Tensor


@dataclass
class HiddenVAEOutput:
    recon_hidden: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor


class HiddenVAE(nn.Module):
    config: HiddenVAEConfig

    def encode(self, hidden: torch.Tensor) -> LatentDistribution:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reparameterize(self, dist: LatentDistribution) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * dist.logvar)
            return dist.mu + torch.randn_like(std) * std
        return dist.mu

    def forward(self, hidden: torch.Tensor) -> HiddenVAEOutput:
        dist = self.encode(hidden)
        z = self.reparameterize(dist)
        recon = self.decode(z)
        return HiddenVAEOutput(
            recon_hidden=recon,
            z=z,
            mu=dist.mu,
            logvar=dist.logvar,
        )

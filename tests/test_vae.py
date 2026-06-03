import pytest
import torch

from chained_flow.vae import HiddenVAEConfig, VAE_REGISTRY, build_hidden_vae


@pytest.mark.parametrize("vae_type", sorted(VAE_REGISTRY))
def test_hidden_vae_architectures_round_trip_shape(vae_type):
    model = build_hidden_vae(
        vae_type,
        HiddenVAEConfig(hidden_size=8, latent_size=3, intermediate_size=5),
    )
    hidden = torch.randn(4, 8)
    output = model(hidden)

    assert output.recon_hidden.shape == hidden.shape
    assert output.mu.shape == (4, 3)
    assert output.logvar.shape == (4, 3)
    assert output.z.shape == (4, 3)


def test_unknown_hidden_vae_type_raises():
    with pytest.raises(ValueError, match="unknown vae_type"):
        build_hidden_vae("missing", HiddenVAEConfig(hidden_size=8, latent_size=3, intermediate_size=5))

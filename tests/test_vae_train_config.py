from pathlib import Path
import importlib.util

from transformers import HfArgumentParser, TrainingArguments


def load_script_module():
    path = Path("scripts/train_vae.py").resolve()
    spec = importlib.util.spec_from_file_location("train_vae", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_smoke_vae_yaml_parses():
    module = load_script_module()
    parser = HfArgumentParser(
        (
            module.VAEModelArguments,
            module.VAEDataArguments,
            module.VAELossArguments,
            TrainingArguments,
        )
    )
    model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
        yaml_file=str(Path("train_configs/smoke_vae.yaml").resolve())
    )

    assert model_args.vae_type == "residual_mlp"
    assert model_args.latent_size == 256
    assert data_args.tokens_per_epoch == 128
    assert loss_args.beta == 0.0001
    assert training_args.output_dir == "checkpoints/hidden-vae-smoke"

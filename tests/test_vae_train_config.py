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
    assert model_args.device == "cuda"
    assert data_args.dataset_path == "sghosts/cf_gsm8k_1k_train"
    assert data_args.dataset_split == "train"
    assert data_args.tokens_per_epoch is None
    assert data_args.validation_fraction == 0.1
    assert loss_args.beta == 0.0001
    assert training_args.output_dir == "/content/drive/MyDrive/chained-flow/vae/ckpts/hidden-vae-smoke"
    assert training_args.per_device_eval_batch_size == 64
    assert training_args.eval_strategy == "epoch"
    assert training_args.save_strategy == "epoch"

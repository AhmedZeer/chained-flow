from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chained_flow.frozen_lm import DEFAULT_MODEL_ID, FrozenLMWrapper
from chained_flow.training.train_vae import HiddenVAETrainingModule, VAELossArguments, VAEModelArguments
from chained_flow.training.vae_dataset import HiddenTokenTensorDataset, TeacherHiddenTokenDataset, collate_hidden_tokens
from chained_flow.training.vae_losses import compute_hidden_vae_loss


@dataclass
class VAEEvalArguments:
    vae_dir: str
    dataset_path: str
    dataset_split: str = "train"
    model_id: str = DEFAULT_MODEL_ID
    device: str | None = None
    dtype: str = "float16"
    batch_size: int = 32
    max_batches: int | None = None
    response_only: bool = True
    local_files_only: bool = False
    output_path: str | None = None


def torch_dtype_from_string(dtype: str | None) -> torch.dtype | None:
    if dtype is None:
        return None
    normalized = dtype.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported dtype: {dtype}")
    return mapping[normalized]


def logit_kl_divergence(
    real_logits: torch.Tensor,
    recon_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if real_logits.shape != recon_logits.shape:
        raise ValueError(
            f"real_logits and recon_logits must have the same shape, got "
            f"{tuple(real_logits.shape)} and {tuple(recon_logits.shape)}"
        )
    real_probs = F.softmax(real_logits.float() / temperature, dim=-1)
    recon_log_probs = F.log_softmax(recon_logits.float() / temperature, dim=-1)
    return F.kl_div(recon_log_probs, real_probs, reduction="batchmean") * (temperature**2)


def load_vae_training_module(vae_dir: str | Path, *, device: torch.device) -> HiddenVAETrainingModule:
    vae_dir = Path(vae_dir)
    config_path = vae_dir / "chained_flow_vae_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing VAE config: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model_args = VAEModelArguments(**config["model_args"])
    loss_args = VAELossArguments(**config["loss_args"])
    module = HiddenVAETrainingModule(model_args, loss_args)

    safetensors_path = vae_dir / "model.safetensors"
    bin_path = vae_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path), device="cpu")
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"missing VAE weights in {vae_dir}")

    module.load_state_dict(state_dict)
    module.to(device)
    module.eval()
    return module


def _mean_metrics(total_metrics: dict[str, float], total_examples: int) -> dict[str, float]:
    return {name: value / max(1, total_examples) for name, value in total_metrics.items()}


@torch.inference_mode()
def evaluate_vae(args: VAEEvalArguments) -> dict[str, Any]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"loading VAE checkpoint: {args.vae_dir}", flush=True)
    vae_module = load_vae_training_module(args.vae_dir, device=device)
    print(f"VAE checkpoint loaded: device={next(vae_module.parameters()).device}", flush=True)

    print(f"loading VAE eval dataset: {args.dataset_path} split={args.dataset_split}", flush=True)
    token_dataset = TeacherHiddenTokenDataset.from_path(
        args.dataset_path,
        split=args.dataset_split,
        response_only=args.response_only,
    )
    eval_dataset = HiddenTokenTensorDataset(token_dataset.hidden_tokens, sample=False)
    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=collate_hidden_tokens)
    print(
        f"VAE eval dataset loaded: tokens={len(eval_dataset)} response_only={args.response_only}",
        flush=True,
    )

    dtype = torch_dtype_from_string(args.dtype)
    print(f"loading LM head: {args.model_id} dtype={args.dtype} device={device}", flush=True)
    wrapper, _ = FrozenLMWrapper.from_pretrained(
        args.model_id,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    lm_head = wrapper.model.lm_head.eval()
    print(f"LM head loaded: device={next(lm_head.parameters()).device}", flush=True)
    lm_head_dtype = next(lm_head.parameters()).dtype

    totals: dict[str, float] = {}
    total_examples = 0
    for step, batch in enumerate(dataloader):
        if args.max_batches is not None and step >= args.max_batches:
            break
        hidden = batch["hidden"].to(device)
        output = vae_module.vae(hidden)
        loss_output = compute_hidden_vae_loss(
            output.recon_hidden,
            hidden,
            mu=output.mu,
            logvar=output.logvar,
            config=vae_module.loss_config,
        )
        real_logits = lm_head(hidden.to(dtype=lm_head_dtype))
        recon_logits = lm_head(output.recon_hidden.to(dtype=lm_head_dtype))
        batch_metrics = {
            **{name: value for name, value in loss_output.components.items()},
            "logit.kl": logit_kl_divergence(real_logits, recon_logits),
            "token.match": (real_logits.argmax(dim=-1) == recon_logits.argmax(dim=-1)).float().mean(),
        }
        batch_size = int(hidden.shape[0])
        total_examples += batch_size
        for name, value in batch_metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size

    metrics = _mean_metrics(totals, total_examples)
    result = {
        "vae_dir": args.vae_dir,
        "dataset_path": args.dataset_path,
        "dataset_split": args.dataset_split,
        "model_id": args.model_id,
        "device": str(device),
        "num_tokens": total_examples,
        "metrics": metrics,
    }
    output_path = Path(args.output_path) if args.output_path else Path(args.vae_dir) / "vae_eval_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    print(f"VAE eval metrics saved: {output_path}", flush=True)
    return result


__all__ = [
    "VAEEvalArguments",
    "evaluate_vae",
    "load_vae_training_module",
    "logit_kl_divergence",
    "torch_dtype_from_string",
]

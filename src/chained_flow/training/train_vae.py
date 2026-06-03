from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import Trainer, TrainingArguments

from chained_flow.training.vae_dataset import TeacherHiddenTokenDataset, collate_hidden_tokens
from chained_flow.training.vae_losses import HiddenVAELossConfig, compute_hidden_vae_loss
from chained_flow.vae import HiddenVAEConfig, build_hidden_vae


@dataclass
class VAEModelArguments:
    vae_type: str = "residual_mlp"
    hidden_size: int = 1024
    latent_size: int = 256
    intermediate_size: int = 512


@dataclass
class VAEDataArguments:
    dataset_path: str = "teacher_states/gsm8k-qwen35-08b-smoke"
    tokens_per_epoch: int | None = None
    token_seed: int = 0
    response_only: bool = True


@dataclass
class VAELossArguments:
    lambda_mse: float = 1.0
    lambda_cos: float = 0.2
    lambda_norm: float = 0.05
    beta: float = 1e-4
    free_bits: float = 0.0


class HiddenVAETrainingModule(nn.Module):
    def __init__(
        self,
        model_args: VAEModelArguments,
        loss_args: VAELossArguments,
    ):
        super().__init__()
        self.vae = build_hidden_vae(
            model_args.vae_type,
            HiddenVAEConfig(
                hidden_size=model_args.hidden_size,
                latent_size=model_args.latent_size,
                intermediate_size=model_args.intermediate_size,
            ),
        )
        self.loss_config = HiddenVAELossConfig(
            lambda_mse=loss_args.lambda_mse,
            lambda_cos=loss_args.lambda_cos,
            lambda_norm=loss_args.lambda_norm,
            beta=loss_args.beta,
            free_bits=loss_args.free_bits,
        )

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.vae(hidden)
        loss_output = compute_hidden_vae_loss(
            output.recon_hidden,
            hidden,
            mu=output.mu,
            logvar=output.logvar,
            config=self.loss_config,
        )
        result: dict[str, torch.Tensor] = {
            "loss": loss_output.total,
            "recon_hidden": output.recon_hidden,
            "z": output.z,
        }
        for name, value in loss_output.components.items():
            result[f"loss_component/{name}"] = value.detach()
        return result


class VAEComponentLoggingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        component_logs = {
            key.replace("loss_component/", ""): float(value.detach().mean().cpu())
            for key, value in outputs.items()
            if key.startswith("loss_component/")
        }
        if component_logs and self.state.global_step % max(1, self.args.logging_steps) == 0:
            self.log(component_logs)
        return (loss, outputs) if return_outputs else loss


def train_vae_with_trainer(
    model_args: VAEModelArguments,
    data_args: VAEDataArguments,
    loss_args: VAELossArguments,
    training_args: TrainingArguments,
) -> dict[str, Any]:
    torch.manual_seed(training_args.seed)
    dataset = TeacherHiddenTokenDataset.from_disk(
        data_args.dataset_path,
        tokens_per_epoch=data_args.tokens_per_epoch,
        seed=data_args.token_seed,
        response_only=data_args.response_only,
    )
    model = HiddenVAETrainingModule(model_args, loss_args)
    trainer = VAEComponentLoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_hidden_tokens,
    )
    train_result = trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))
    trainer.save_model(training_args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chained_flow_vae_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_args": asdict(model_args),
                "data_args": asdict(data_args),
                "loss_args": asdict(loss_args),
            },
            f,
            indent=2,
        )
    return {
        "output_dir": training_args.output_dir,
        "global_step": trainer.state.global_step,
        "metrics": train_result.metrics,
    }

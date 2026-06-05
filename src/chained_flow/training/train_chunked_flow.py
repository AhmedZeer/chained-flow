from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments

from chained_flow.context import ChainedFlowContext
from chained_flow.drafters.chunked_flow import SingleExpertFlowConfig, SingleExpertFlowDrafter
from chained_flow.frozen_lm import DEFAULT_MODEL_ID, FrozenLMWrapper
from chained_flow.training.collators import collate_teacher_windows
from chained_flow.training.window_dataset import TeacherWindowDataset


@dataclass
class ChunkedFlowModelArguments:
    model_id: str = DEFAULT_MODEL_ID
    context_size: int = 4
    draft_length: int = 2
    chunk_size: int = 2
    vae_dir: str | None = None
    expert_dim: int = 128
    num_heads: int = 4
    ffn_multiplier: int = 4
    num_flow_steps: int = 1
    noise_scale: float = 1.0
    local_files_only: bool = False
    device: str | None = None


@dataclass
class TeacherDataArguments:
    dataset_path: str = "teacher_states/gsm8k-qwen35-08b-smoke"
    dataset_split: str = "train"
    windows_per_epoch: int | None = None
    window_seed: int = 0


@dataclass
class FlowLossArguments:
    lambda_flow: float = 1.0
    lambda_latent: float = 0.5
    lambda_hidden: float = 0.5
    lambda_cos: float = 0.1
    lambda_ce: float = 0.2
    lambda_accept: float = 0.1
    gamma: float = 0.8
    eps: float = 1e-8


@dataclass
class FlowLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


class SingleExpertFlowTrainingModule(nn.Module):
    def __init__(
        self,
        frozen_lm: FrozenLMWrapper,
        drafter_config: SingleExpertFlowConfig,
        loss_config: FlowLossArguments,
    ):
        super().__init__()
        self.drafter = SingleExpertFlowDrafter(frozen_lm, drafter_config)
        self.loss_config = loss_config
        lm_head = frozen_lm.model.lm_head
        self.register_buffer("lm_head_weight", lm_head.weight.detach().clone(), persistent=False)
        bias = getattr(lm_head, "bias", None)
        if bias is None:
            self.lm_head_bias = None
        else:
            self.register_buffer("lm_head_bias", bias.detach().clone(), persistent=False)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        return {key: value for key, value in state.items() if not key.startswith("drafter.vae.")}

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        full_state = super().state_dict()
        full_state.update(state_dict)
        return super().load_state_dict(full_state, strict=strict, assign=assign)

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.matmul(hidden_states, self.lm_head_weight.t())
        if self.lm_head_bias is not None:
            logits = logits + self.lm_head_bias
        return logits

    def _relative_hidden_mse(self, pred_hidden: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred_hidden, target_hidden, reduction="none").mean(dim=-1)
        target_power = target_hidden.float().pow(2).mean(dim=-1).clamp_min(self.loss_config.eps)
        return (mse.float() / target_power).mean()

    def _hidden_cosine_loss(self, pred_hidden: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
        pred = pred_hidden.reshape(-1, pred_hidden.shape[-1])
        target = target_hidden.reshape(-1, target_hidden.shape[-1])
        pred_norm = pred.norm(dim=-1)
        target_norm = target.norm(dim=-1)
        both_zero = (pred_norm == 0) & (target_norm == 0)
        cosine = F.cosine_similarity(pred, target, dim=-1)
        cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
        return (1.0 - cosine).mean()

    def _token_ce_loss(self, logits: torch.Tensor, future_tokens: torch.Tensor) -> torch.Tensor:
        batch, length, vocab = logits.shape
        return F.cross_entropy(logits.reshape(batch * length, vocab), future_tokens.reshape(batch * length))

    def _expected_acceptance_loss(self, logits: torch.Tensor, future_tokens: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        token_probs = probs.gather(dim=-1, index=future_tokens.unsqueeze(-1)).squeeze(-1)
        prefix_probs = torch.cumprod(token_probs.clamp_min(self.loss_config.eps), dim=1)
        weights = self.loss_config.gamma ** torch.arange(logits.shape[1], device=logits.device, dtype=prefix_probs.dtype)
        return -(prefix_probs * weights.unsqueeze(0)).sum(dim=1).mean()

    def compute_flow_loss(
        self,
        context_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        future_tokens: torch.Tensor,
    ) -> tuple[FlowLossOutput, torch.Tensor, torch.Tensor]:
        if target_hidden.shape[1] != self.drafter.config.draft_length:
            raise ValueError(
                f"target_hidden must have draft length {self.drafter.config.draft_length}, got {target_hidden.shape[1]}"
            )
        z_ctx = self.drafter.encode_hidden(context_hidden)
        z_target = self.drafter.encode_hidden(target_hidden)
        z0 = torch.randn_like(z_target) * self.drafter.config.noise_scale
        batch = z_target.shape[0]
        tau = torch.rand(batch, 1, 1, device=z_target.device, dtype=z_target.dtype)
        z_tau = (1.0 - tau) * z0 + tau * z_target
        v_star = z_target - z0
        v_pred = self.drafter.expert(z_tau, tau, z_ctx)

        components: dict[str, torch.Tensor] = {}
        components["flow.mse"] = F.mse_loss(v_pred, v_star)

        z_pred = self.drafter.integrate_latents(z_ctx, z0=z0)
        components["latent.mse"] = F.mse_loss(z_pred, z_target)

        pred_hidden = self.drafter.decode_latent(z_pred)
        components["hidden.rel_mse"] = self._relative_hidden_mse(pred_hidden, target_hidden)
        components["hidden.cos"] = self._hidden_cosine_loss(pred_hidden, target_hidden)

        logits = self.lm_head(pred_hidden)
        components["logit.ce"] = self._token_ce_loss(logits, future_tokens)
        components["verifier.expected_accept"] = self._expected_acceptance_loss(logits, future_tokens)

        total = pred_hidden.new_zeros(())
        total = total + self.loss_config.lambda_flow * components["flow.mse"]
        total = total + self.loss_config.lambda_latent * components["latent.mse"]
        total = total + self.loss_config.lambda_hidden * components["hidden.rel_mse"]
        total = total + self.loss_config.lambda_cos * components["hidden.cos"]
        total = total + self.loss_config.lambda_ce * components["logit.ce"]
        total = total + self.loss_config.lambda_accept * components["verifier.expected_accept"]
        return FlowLossOutput(total=total, components=components), z_pred, pred_hidden

    def forward(
        self,
        context_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        future_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_output, pred_latent, pred_hidden = self.compute_flow_loss(context_hidden, target_hidden, future_tokens)
        output: dict[str, torch.Tensor] = {
            "loss": loss_output.total,
            "pred_latent": pred_latent,
            "pred_hidden": pred_hidden,
        }
        for name, value in loss_output.components.items():
            output[f"loss_component/{name}"] = value.detach()
        return output


class ComponentLoggingTrainer(Trainer):
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


def flow_config_from_args(args: ChunkedFlowModelArguments) -> SingleExpertFlowConfig:
    return SingleExpertFlowConfig(
        context_size=args.context_size,
        draft_length=args.draft_length,
        chunk_size=args.chunk_size,
        vae_dir=args.vae_dir,
        expert_dim=args.expert_dim,
        num_heads=args.num_heads,
        ffn_multiplier=args.ffn_multiplier,
        num_flow_steps=args.num_flow_steps,
        noise_scale=args.noise_scale,
    )


def train_chunked_flow_with_trainer(
    model_args: ChunkedFlowModelArguments,
    data_args: TeacherDataArguments,
    loss_args: FlowLossArguments,
    training_args: TrainingArguments,
) -> dict[str, Any]:
    torch.manual_seed(training_args.seed)
    context = ChainedFlowContext.from_pretrained(
        model_args.model_id,
        device=model_args.device,
        local_files_only=model_args.local_files_only,
    )
    frozen_lm = context.frozen_lm

    dataset = TeacherWindowDataset.from_path(
        data_args.dataset_path,
        split=data_args.dataset_split,
        context_size=model_args.context_size,
        draft_length=model_args.draft_length,
        windows_per_epoch=data_args.windows_per_epoch,
        seed=data_args.window_seed,
    )

    model = SingleExpertFlowTrainingModule(
        frozen_lm,
        flow_config_from_args(model_args),
        loss_args,
    )
    trainer = ComponentLoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_teacher_windows,
    )
    train_result = trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))
    trainer.save_model(training_args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chained_flow_chunked_flow_config.json").open("w", encoding="utf-8") as f:
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


__all__ = [
    "ChunkedFlowModelArguments",
    "ComponentLoggingTrainer",
    "FlowLossArguments",
    "FlowLossOutput",
    "SingleExpertFlowTrainingModule",
    "TeacherDataArguments",
    "flow_config_from_args",
    "train_chunked_flow_with_trainer",
]

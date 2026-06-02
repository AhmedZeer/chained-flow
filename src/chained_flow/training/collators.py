from __future__ import annotations

import torch


def collate_teacher_windows(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "context_hidden": torch.stack([item["context_hidden"] for item in batch], dim=0),
        "target_hidden": torch.stack([item["target_hidden"] for item in batch], dim=0),
        "future_tokens": torch.stack([item["future_tokens"] for item in batch], dim=0),
    }

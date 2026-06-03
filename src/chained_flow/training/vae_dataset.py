from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from datasets import Dataset, load_from_disk
import torch
from torch.utils.data import Dataset as TorchDataset


@dataclass(frozen=True)
class TeacherHiddenToken:
    hidden: torch.Tensor


class TeacherHiddenTokenDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
        *,
        tokens_per_epoch: int | None = None,
        seed: int = 0,
        response_only: bool = True,
    ):
        self.dataset = dataset
        self.seed = seed
        self.response_only = response_only
        self.valid_rows: list[tuple[int, int, int]] = []
        total_tokens = 0
        for row_idx, row in enumerate(dataset):
            length = int(row["num_tokens"])
            start = int(row.get("prompt_length", 0)) if response_only else 0
            end = length
            if end > start:
                self.valid_rows.append((row_idx, start, end))
                total_tokens += end - start
        if not self.valid_rows:
            raise ValueError("dataset contains no hidden-token rows for VAE training")
        self.tokens_per_epoch = tokens_per_epoch or total_tokens

    @classmethod
    def from_disk(
        cls,
        path: str,
        *,
        tokens_per_epoch: int | None = None,
        seed: int = 0,
        response_only: bool = True,
    ) -> "TeacherHiddenTokenDataset":
        return cls(
            load_from_disk(path),
            tokens_per_epoch=tokens_per_epoch,
            seed=seed,
            response_only=response_only,
        )

    def __len__(self) -> int:
        return self.tokens_per_epoch

    def _sample_row_and_position(self, index: int) -> tuple[dict[str, Any], int]:
        rng = random.Random(self.seed + index)
        row_idx, start, end = rng.choice(self.valid_rows)
        return self.dataset[int(row_idx)], rng.randrange(start, end)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row, position = self._sample_row_and_position(index)
        hidden = torch.tensor(row["final_hidden"][position], dtype=torch.float32)
        return {"hidden": hidden}


def collate_hidden_tokens(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {"hidden": torch.stack([item["hidden"] for item in batch], dim=0)}

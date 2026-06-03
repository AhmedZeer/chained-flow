from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from datasets import Dataset, load_dataset, load_from_disk
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
        lengths = dataset["num_tokens"]
        prompt_lengths = dataset["prompt_length"] if response_only and "prompt_length" in dataset.column_names else None
        for row_idx, length_value in enumerate(lengths):
            length = int(length_value)
            start = int(prompt_lengths[row_idx]) if prompt_lengths is not None else 0
            end = length
            if end > start:
                self.valid_rows.append((row_idx, start, end))
        if not self.valid_rows:
            raise ValueError("dataset contains no hidden-token rows for VAE training")
        self.hidden_tokens = self._flatten_hidden_tokens(dataset)
        self.tokens_per_epoch = tokens_per_epoch or int(self.hidden_tokens.shape[0])

    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        split: str = "train",
        tokens_per_epoch: int | None = None,
        seed: int = 0,
        response_only: bool = True,
    ) -> "TeacherHiddenTokenDataset":
        dataset = load_from_disk(path) if Path(path).exists() else load_dataset(path, split=split)
        return cls(
            dataset,
            tokens_per_epoch=tokens_per_epoch,
            seed=seed,
            response_only=response_only,
        )

    @classmethod
    def from_disk(
        cls,
        path: str,
        *,
        tokens_per_epoch: int | None = None,
        seed: int = 0,
        response_only: bool = True,
    ) -> "TeacherHiddenTokenDataset":
        return cls.from_path(
            path,
            tokens_per_epoch=tokens_per_epoch,
            seed=seed,
            response_only=response_only,
        )

    def __len__(self) -> int:
        return self.tokens_per_epoch

    def _flatten_hidden_tokens(self, dataset: Dataset) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        hidden_column = dataset["final_hidden"]
        for row_idx, start, end in self.valid_rows:
            hidden = torch.as_tensor(hidden_column[row_idx][start:end], dtype=torch.float32)
            if hidden.ndim != 2:
                raise ValueError(f"final_hidden row {row_idx} must have shape [tokens, hidden_size]")
            chunks.append(hidden)
        return torch.cat(chunks, dim=0).contiguous()

    def _sample_token_index(self, index: int) -> int:
        rng = random.Random(self.seed + index)
        return rng.randrange(int(self.hidden_tokens.shape[0]))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"hidden": self.hidden_tokens[self._sample_token_index(index)]}


def collate_hidden_tokens(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {"hidden": torch.stack([item["hidden"] for item in batch], dim=0)}

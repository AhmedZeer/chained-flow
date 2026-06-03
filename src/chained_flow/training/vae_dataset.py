from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from datasets import Dataset, load_dataset, load_from_disk
import torch
from torch.utils.data import Dataset as TorchDataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is normally present through datasets/transformers.
    tqdm = None


@dataclass(frozen=True)
class TeacherHiddenToken:
    hidden: torch.Tensor


class HiddenTokenTensorDataset(TorchDataset):
    def __init__(
        self,
        hidden_tokens: torch.Tensor,
        *,
        tokens_per_epoch: int | None = None,
        seed: int = 0,
        sample: bool = True,
    ):
        if hidden_tokens.ndim != 2:
            raise ValueError("hidden_tokens must have shape [tokens, hidden_size]")
        if hidden_tokens.shape[0] == 0:
            raise ValueError("hidden_tokens must contain at least one token")
        self.hidden_tokens = hidden_tokens.contiguous()
        self.tokens_per_epoch = tokens_per_epoch or int(hidden_tokens.shape[0])
        self.seed = seed
        self.sample = sample

    def __len__(self) -> int:
        return self.tokens_per_epoch if self.sample else int(self.hidden_tokens.shape[0])

    def _token_index(self, index: int) -> int:
        if not self.sample:
            return index
        rng = random.Random(self.seed + index)
        return rng.randrange(int(self.hidden_tokens.shape[0]))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"hidden": self.hidden_tokens[self._token_index(index)]}


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
        self.requested_tokens_per_epoch = tokens_per_epoch
        self.valid_rows: list[tuple[int, int, int]] = []
        print("formatting VAE dataset: scanning hidden-token rows", flush=True)
        lengths = dataset["num_tokens"]
        prompt_lengths = dataset["prompt_length"] if response_only and "prompt_length" in dataset.column_names else None
        iterator = enumerate(lengths)
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=len(lengths),
                desc="scanning VAE rows",
                unit="row",
            )
        for row_idx, length_value in iterator:
            length = int(length_value)
            start = int(prompt_lengths[row_idx]) if prompt_lengths is not None else 0
            end = length
            if end > start:
                self.valid_rows.append((row_idx, start, end))
        if not self.valid_rows:
            raise ValueError("dataset contains no hidden-token rows for VAE training")
        self.hidden_tokens = self._flatten_hidden_tokens(dataset)
        print(
            f"VAE dataset formatted: hidden_tokens={self.hidden_tokens.shape[0]} "
            f"hidden_size={self.hidden_tokens.shape[1]}",
            flush=True,
        )
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
        iterator = self.valid_rows
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=len(self.valid_rows),
                desc="flattening VAE hidden states",
                unit="row",
            )
        for row_idx, start, end in iterator:
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

    def train_val_split(
        self,
        *,
        val_fraction: float = 0.1,
        seed: int = 0,
    ) -> tuple[HiddenTokenTensorDataset, HiddenTokenTensorDataset]:
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        total = int(self.hidden_tokens.shape[0])
        val_count = max(1, int(total * val_fraction))
        if val_count >= total:
            raise ValueError("not enough hidden tokens to create a non-empty train/val split")
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(total, generator=generator)
        val_indices = permutation[:val_count]
        train_indices = permutation[val_count:]
        train_tokens_per_epoch = self.requested_tokens_per_epoch or int(train_indices.numel())
        train_dataset = HiddenTokenTensorDataset(
            self.hidden_tokens[train_indices],
            tokens_per_epoch=train_tokens_per_epoch,
            seed=self.seed,
            sample=True,
        )
        val_dataset = HiddenTokenTensorDataset(
            self.hidden_tokens[val_indices],
            sample=False,
        )
        return train_dataset, val_dataset


def collate_hidden_tokens(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {"hidden": torch.stack([item["hidden"] for item in batch], dim=0)}

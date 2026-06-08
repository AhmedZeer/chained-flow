from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk
import torch
from torch.utils.data import Dataset as TorchDataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is normally present through datasets/transformers.
    tqdm = None


@dataclass(frozen=True)
class TeacherWindow:
    context_hidden: torch.Tensor
    target_hidden: torch.Tensor
    future_tokens: torch.Tensor


class TeacherWindowDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
        *,
        context_size: int,
        draft_length: int,
        windows_per_epoch: int | None = None,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.context_size = context_size
        self.draft_length = draft_length
        self.seed = seed
        self.valid_rows: list[tuple[int, int, int]] = []
        print("formatting flow dataset: scanning teacher-window rows", flush=True)
        lengths = dataset["num_tokens"]
        prompt_lengths = dataset["prompt_length"] if "prompt_length" in dataset.column_names else None
        iterator = enumerate(lengths)
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=len(lengths),
                desc="scanning flow rows",
                unit="row",
            )
        for row_idx, length_value in iterator:
            length = int(length_value)
            prompt_length = int(prompt_lengths[row_idx]) if prompt_lengths is not None else 1
            min_t = max(0, prompt_length - 1)
            max_t = length - draft_length - 1
            if max_t >= min_t:
                self.valid_rows.append((row_idx, min_t, max_t))
        if not self.valid_rows:
            raise ValueError("dataset contains no response-side rows long enough for the requested draft_length")
        self.available_windows = sum(max_t - min_t + 1 for _, min_t, max_t in self.valid_rows)
        self.windows_per_epoch = windows_per_epoch or self.available_windows
        print(
            f"flow dataset formatted: valid_rows={len(self.valid_rows)} "
            f"available_windows={self.available_windows} windows_per_epoch={self.windows_per_epoch}",
            flush=True,
        )

    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        split: str = "train",
        context_size: int,
        draft_length: int,
        windows_per_epoch: int | None = None,
        seed: int = 0,
    ) -> "TeacherWindowDataset":
        if Path(path).exists():
            print(f"loading flow dataset from disk: {path}", flush=True)
            dataset = load_from_disk(path)
        else:
            print(f"loading flow dataset from hub: {path} split={split}", flush=True)
            dataset = load_dataset(path, split=split)
        print(f"flow dataset source loaded: rows={len(dataset)} columns={dataset.column_names}", flush=True)
        return cls(
            dataset,
            context_size=context_size,
            draft_length=draft_length,
            windows_per_epoch=windows_per_epoch,
            seed=seed,
        )

    @classmethod
    def from_disk(
        cls,
        path: str,
        *,
        context_size: int,
        draft_length: int,
        windows_per_epoch: int | None = None,
        seed: int = 0,
    ) -> "TeacherWindowDataset":
        return cls.from_path(
            path,
            context_size=context_size,
            draft_length=draft_length,
            windows_per_epoch=windows_per_epoch,
            seed=seed,
        )

    def __len__(self) -> int:
        return self.windows_per_epoch

    def _sample_row_and_t(self, index: int) -> tuple[dict[str, Any], int]:
        rng = random.Random(self.seed + index)
        row_idx, min_t, max_t = rng.choice(self.valid_rows)
        return self.dataset[int(row_idx)], rng.randint(min_t, max_t)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row, t = self._sample_row_and_t(index)
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        hidden = torch.tensor(row["final_hidden"], dtype=torch.float32)

        context_start = t - self.context_size + 1
        if context_start >= 0:
            context_hidden = hidden[context_start : t + 1]
        else:
            pad = hidden[:1].expand(-context_start, -1)
            context_hidden = torch.cat([pad, hidden[: t + 1]], dim=0)

        target_hidden = hidden[t : t + self.draft_length]
        future_tokens = input_ids[t + 1 : t + self.draft_length + 1]

        return {
            "context_hidden": context_hidden,
            "target_hidden": target_hidden,
            "future_tokens": future_tokens,
        }

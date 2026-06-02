from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from datasets import Dataset, load_from_disk
import torch
from torch.utils.data import Dataset as TorchDataset


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
        for row_idx, row in enumerate(dataset):
            length = int(row["num_tokens"])
            prompt_length = int(row.get("prompt_length", 1))
            min_t = max(0, prompt_length - 1)
            max_t = length - draft_length - 1
            if max_t >= min_t:
                self.valid_rows.append((row_idx, min_t, max_t))
        if not self.valid_rows:
            raise ValueError("dataset contains no response-side rows long enough for the requested draft_length")
        self.windows_per_epoch = windows_per_epoch or sum(max_t - min_t + 1 for _, min_t, max_t in self.valid_rows)

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
        return cls(
            load_from_disk(path),
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

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
        materialize_rows: bool = True,
    ):
        self.dataset = dataset
        self.context_size = context_size
        self.draft_length = draft_length
        self.seed = seed
        self.materialize_rows = materialize_rows
        self.valid_rows: list[tuple[int, int, int]] = []
        self.row_tensors: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
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
        if materialize_rows:
            self.row_tensors = self._materialize_valid_rows(dataset)

    def _materialize_valid_rows(self, dataset: Dataset) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        print("materializing flow dataset rows: caching input_ids and final_hidden tensors", flush=True)
        input_ids_column = dataset["input_ids"]
        hidden_column = dataset["final_hidden"]
        row_tensors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        total_tokens = 0
        total_hidden_elements = 0
        iterator = self.valid_rows
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=len(self.valid_rows),
                desc="materializing flow rows",
                unit="row",
            )
        for row_idx, _, _ in iterator:
            input_ids = torch.as_tensor(input_ids_column[row_idx], dtype=torch.long).contiguous()
            hidden = torch.as_tensor(hidden_column[row_idx], dtype=torch.float32).contiguous()
            if hidden.ndim != 2:
                raise ValueError(f"final_hidden row {row_idx} must have shape [tokens, hidden_size]")
            if input_ids.ndim != 1:
                raise ValueError(f"input_ids row {row_idx} must have shape [tokens]")
            if input_ids.shape[0] < hidden.shape[0]:
                raise ValueError(f"input_ids row {row_idx} is shorter than final_hidden")
            row_tensors[int(row_idx)] = (input_ids, hidden)
            total_tokens += int(hidden.shape[0])
            total_hidden_elements += int(hidden.numel())
        approx_hidden_gib = total_hidden_elements * 4 / 1024**3
        print(
            f"flow rows materialized: rows={len(row_tensors)} tokens={total_tokens} "
            f"hidden_gib_float32={approx_hidden_gib:.2f}",
            flush=True,
        )
        return row_tensors

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
        materialize_rows: bool = True,
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
            materialize_rows=materialize_rows,
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
        materialize_rows: bool = True,
    ) -> "TeacherWindowDataset":
        return cls.from_path(
            path,
            context_size=context_size,
            draft_length=draft_length,
            windows_per_epoch=windows_per_epoch,
            seed=seed,
            materialize_rows=materialize_rows,
        )

    def __len__(self) -> int:
        return self.windows_per_epoch

    def _sample_row_and_t(self, index: int) -> tuple[int, int]:
        rng = random.Random(self.seed + index)
        row_idx, min_t, max_t = rng.choice(self.valid_rows)
        return int(row_idx), rng.randint(min_t, max_t)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row_idx, t = self._sample_row_and_t(index)
        if self.row_tensors is not None:
            input_ids, hidden = self.row_tensors[row_idx]
        else:
            row = self.dataset[row_idx]
            input_ids = torch.as_tensor(row["input_ids"], dtype=torch.long)
            hidden = torch.as_tensor(row["final_hidden"], dtype=torch.float32)

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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from datasets import Dataset, Features, Sequence, Value, load_dataset
import torch

from chained_flow.context import ChainedFlowContext
from chained_flow.frozen_lm import DEFAULT_MODEL_ID
from chained_flow.timing import TimingStats, timed_section


@dataclass(frozen=True)
class TeacherCollectionConfig:
    model_id: str = DEFAULT_MODEL_ID
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    split: str = "train"
    source: str = "gsm8k"
    format_name: str = "qwen_chat_qa"
    limit: int | None = None
    max_tokens: int | None = None
    generation_max_new_tokens: int = 256
    storage_dtype: str = "float32"
    local_files_only: bool = False


def teacher_dataset_features() -> Features:
    return Features(
        {
            "text": Value("string"),
            "input_ids": Sequence(Value("int32")),
            "final_hidden": Sequence(Sequence(Value("float32"))),
            "example_id": Value("string"),
            "source": Value("string"),
            "split": Value("string"),
            "format_name": Value("string"),
            "model_id": Value("string"),
            "hidden_dtype": Value("string"),
            "num_tokens": Value("int32"),
        }
    )


def format_gsm8k_prompt(example: dict[str, Any], tokenizer: Any) -> str:
    question = example["question"].strip()
    messages = [{"role": "user", "content": question}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Question:\n{question}\n\nAnswer:\n"


def _storage_torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError("storage_dtype must be one of: float32, float16, bfloat16")


def _iter_limited(dataset: Iterable[dict[str, Any]], limit: int | None) -> Iterable[tuple[int, dict[str, Any]]]:
    for index, example in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        yield index, example


@torch.inference_mode()
def collect_teacher_dataset(config: TeacherCollectionConfig) -> tuple[Dataset, TimingStats]:
    timings = TimingStats()
    context = ChainedFlowContext.from_pretrained(
        config.model_id,
        local_files_only=config.local_files_only,
    )
    timings.merge(context.timings)
    wrapper = context.frozen_lm
    storage_dtype = _storage_torch_dtype(config.storage_dtype)

    with timed_section(timings, "dataset_load"):
        raw_dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split=config.split,
        )

    rows: list[dict[str, Any]] = []
    with timed_section(timings, "teacher_collection", wrapper.device):
        for index, example in _iter_limited(raw_dataset, config.limit):
            prompt_text = format_gsm8k_prompt(example, wrapper.tokenizer)
            prompt_ids = wrapper.tokenize(prompt_text)
            generated_ids = wrapper.model.generate(
                input_ids=prompt_ids,
                max_new_tokens=config.generation_max_new_tokens,
                do_sample=False,
                pad_token_id=wrapper.eos_token_id,
            )
            input_ids = generated_ids.to(wrapper.device)
            if config.max_tokens is not None:
                input_ids = input_ids[:, : config.max_tokens]
            if input_ids.shape[1] < 2:
                continue

            state, prefill_timings = wrapper.prefill(input_ids)
            timings.merge(prefill_timings)
            hidden = state.final_hidden[0].detach().to(storage_dtype).cpu()
            text = wrapper.decode(input_ids[0], skip_special_tokens=False)
            rows.append(
                {
                    "text": text,
                    "input_ids": input_ids[0].detach().cpu().to(torch.int32).tolist(),
                    "final_hidden": hidden.to(torch.float32).tolist(),
                    "example_id": str(example.get("id", index)),
                    "source": config.source,
                    "split": config.split,
                    "format_name": config.format_name,
                    "model_id": config.model_id,
                    "hidden_dtype": config.storage_dtype,
                    "num_tokens": int(input_ids.shape[1]),
                }
            )

    with timed_section(timings, "dataset_build"):
        dataset = Dataset.from_list(rows, features=teacher_dataset_features())
    return dataset, timings

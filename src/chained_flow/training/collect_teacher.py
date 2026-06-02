from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence as TypingSequence

from datasets import Dataset, Features, Sequence, Value, load_dataset
import torch
from tqdm.auto import tqdm

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
    batch_size: int = 1
    storage_dtype: str = "float32"
    local_files_only: bool = False


def teacher_dataset_features() -> Features:
    return Features(
        {
            "text": Value("string"),
            "prompt_text": Value("string"),
            "generated_text": Value("string"),
            "input_ids": Sequence(Value("int32")),
            "final_hidden": Sequence(Sequence(Value("float32"))),
            "example_id": Value("string"),
            "source": Value("string"),
            "split": Value("string"),
            "format_name": Value("string"),
            "model_id": Value("string"),
            "hidden_dtype": Value("string"),
            "num_tokens": Value("int32"),
            "prompt_length": Value("int32"),
        }
    )


def format_gsm8k_prompt(example: dict[str, Any], tokenizer: Any) -> str:
    question = example["question"].strip()
    messages = [{"role": "user", "content": question}]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
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


def _batched(items: Iterable[tuple[int, dict[str, Any]]], batch_size: int) -> Iterable[list[tuple[int, dict[str, Any]]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[tuple[int, dict[str, Any]]] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _ensure_padding(tokenizer: Any, eos_token_id: int | None) -> int:
    tokenizer.padding_side = "left"
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    if eos_token_id is None:
        raise ValueError("tokenizer has no pad_token_id and model has no eos_token_id")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = eos_token_id
    return int(eos_token_id)


def _eos_token_ids(tokenizer: Any, eos_token_id: int | list[int] | None) -> list[int]:
    ids: list[int] = []
    if isinstance(eos_token_id, int):
        ids.append(eos_token_id)
    elif isinstance(eos_token_id, list):
        ids.extend(int(item) for item in eos_token_id)

    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            ids.append(token_id)

    return sorted(set(ids))


def _sequence_spans(
    input_ids: torch.Tensor,
    pad_token_id: int,
    eos_token_id: int | list[int] | None,
    prompt_lengths: TypingSequence[int],
) -> list[tuple[int, int]]:
    eos_ids = set(eos_token_id if isinstance(eos_token_id, list) else ([] if eos_token_id is None else [eos_token_id]))
    spans: list[tuple[int, int]] = []
    for row, prompt_length in zip(input_ids, prompt_lengths, strict=True):
        nonpad = (row != pad_token_id).nonzero(as_tuple=False)
        if not len(nonpad):
            spans.append((0, 0))
            continue
        start = int(nonpad[0].item())
        end = int(nonpad[-1].item() + 1)
        prompt_end = min(start + int(prompt_length), end)
        if eos_ids:
            eos_positions = torch.tensor(
                [offset for offset, token_id in enumerate(row[prompt_end:end].tolist()) if token_id in eos_ids],
                device=row.device,
            )
            if len(eos_positions):
                end = prompt_end + int(eos_positions[0].item() + 1)
        spans.append((start, end))
    return spans


def _left_pad_sequences(
    sequences: TypingSequence[torch.Tensor],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(seq.shape[0] for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    for row_idx, seq in enumerate(sequences):
        seq = seq.to(device)
        input_ids[row_idx, -seq.shape[0] :] = seq
        attention_mask[row_idx, -seq.shape[0] :] = 1
    return input_ids, attention_mask


@torch.inference_mode()
def collect_teacher_dataset(config: TeacherCollectionConfig) -> tuple[Dataset, TimingStats]:
    timings = TimingStats()
    print(f"loading model: {config.model_id}", flush=True)
    context = ChainedFlowContext.from_pretrained(
        config.model_id,
        local_files_only=config.local_files_only,
    )
    timings.merge(context.timings)
    wrapper = context.frozen_lm
    storage_dtype = _storage_torch_dtype(config.storage_dtype)
    pad_token_id = _ensure_padding(wrapper.tokenizer, wrapper.eos_token_id)
    eos_token_ids = _eos_token_ids(wrapper.tokenizer, wrapper.eos_token_id)
    print(f"model loaded: {config.model_id}", flush=True)

    print(
        f"loading dataset: {config.dataset_name}/{config.dataset_config} split={config.split}",
        flush=True,
    )
    with timed_section(timings, "dataset_load"):
        raw_dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split=config.split,
        )
    print(f"dataset loaded: rows={len(raw_dataset)}", flush=True)

    rows: list[dict[str, Any]] = []
    with timed_section(timings, "teacher_collection", wrapper.device):
        total = min(config.limit, len(raw_dataset)) if config.limit is not None else len(raw_dataset)
        examples = tqdm(
            _iter_limited(raw_dataset, config.limit),
            total=total,
            desc="collecting teacher states",
        )
        for batch in _batched(examples, config.batch_size):
            batch_indices = [index for index, _ in batch]
            batch_examples = [example for _, example in batch]
            prompt_texts = [format_gsm8k_prompt(example, wrapper.tokenizer) for example in batch_examples]
            encoded = wrapper.tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
            )
            prompt_ids = encoded.input_ids.to(wrapper.device)
            prompt_attention_mask = encoded.attention_mask.to(wrapper.device)
            prompt_lengths = prompt_attention_mask.sum(dim=1).tolist()
            generated_ids = wrapper.model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=config.generation_max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_ids,
            )
            input_ids = generated_ids.to(wrapper.device)
            if config.max_tokens is not None:
                input_ids = input_ids[:, : config.max_tokens]
            spans = _sequence_spans(input_ids, pad_token_id, eos_token_ids, prompt_lengths)
            trimmed_sequences = [
                input_ids[row_idx, start:end]
                for row_idx, (start, end) in enumerate(spans)
                if end - start >= 2
            ]
            if not trimmed_sequences:
                continue

            padded_ids, attention_mask = _left_pad_sequences(
                trimmed_sequences,
                pad_token_id=pad_token_id,
                device=wrapper.device,
            )
            with timed_section(timings, "prefill", wrapper.device):
                outputs = wrapper.model(
                    input_ids=padded_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
            )
            final_hidden = outputs.hidden_states[-1]
            kept_row_idx = 0
            for row_idx, (start, end) in enumerate(spans):
                length = end - start
                if length < 2:
                    continue
                input_row = trimmed_sequences[kept_row_idx]
                left_pad = padded_ids.shape[1] - input_row.shape[0]
                hidden = final_hidden[kept_row_idx, left_pad : left_pad + input_row.shape[0]]
                prompt_length = min(int(prompt_lengths[row_idx]), input_row.shape[0])
                text = wrapper.decode(input_row, skip_special_tokens=False)
                prompt_text = wrapper.decode(input_row[:prompt_length], skip_special_tokens=False)
                generated_text = wrapper.decode(input_row[prompt_length:], skip_special_tokens=False)
                rows.append(
                    {
                        "text": text,
                        "prompt_text": prompt_text,
                        "generated_text": generated_text,
                        "input_ids": input_row.detach().cpu().to(torch.int32).tolist(),
                        "final_hidden": hidden.detach().to(storage_dtype).cpu().to(torch.float32).tolist(),
                        "example_id": str(batch_examples[row_idx].get("id", batch_indices[row_idx])),
                        "source": config.source,
                        "split": config.split,
                        "format_name": config.format_name,
                        "model_id": config.model_id,
                        "hidden_dtype": config.storage_dtype,
                        "num_tokens": int(input_row.shape[0]),
                        "prompt_length": int(prompt_length),
                    }
                )
                kept_row_idx += 1

    with timed_section(timings, "dataset_build"):
        dataset = Dataset.from_list(rows, features=teacher_dataset_features())
    return dataset, timings

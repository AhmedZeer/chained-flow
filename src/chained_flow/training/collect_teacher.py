from __future__ import annotations

from dataclasses import dataclass
import gc
import inspect
from typing import Any, Iterable, Sequence as TypingSequence

from datasets import Dataset, Features, Sequence, Value, load_dataset
import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer

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
    device: str | None = None
    dtype: str | None = None
    seed: int = 0
    vllm_max_model_len: int | None = None
    tmp_output_dir: str | None = None
    tmp_push_to_hub: str | None = None
    private: bool = False


def _hidden_feature_dtype(storage_dtype: str) -> str:
    if storage_dtype == "float16":
        return "float16"
    if storage_dtype == "float32":
        return "float32"
    raise ValueError("HF dataset hidden storage supports float32 or float16")


def teacher_dataset_features(storage_dtype: str = "float32") -> Features:
    hidden_dtype = _hidden_feature_dtype(storage_dtype)
    return Features(
        {
            "text": Value("string"),
            "prompt_text": Value("string"),
            "generated_text": Value("string"),
            "input_ids": Sequence(Value("int32")),
            "final_hidden": Sequence(Sequence(Value(hidden_dtype))),
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


def teacher_answer_dataset_features() -> Features:
    return Features(
        {
            "text": Value("string"),
            "prompt_text": Value("string"),
            "generated_text": Value("string"),
            "input_ids": Sequence(Value("int32")),
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
    raise ValueError("storage_dtype must be one of: float32, float16")


def _model_torch_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    raise ValueError("dtype must be one of: float32, float16")


def _backbone(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "base_model"):
        return model.base_model
    raise AttributeError("could not find a backbone module on the causal LM")


def _load_vllm():
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vLLM is required for offline teacher answer generation. Install vllm in the "
            "collection environment before running collect_teacher_states.py."
        ) from exc
    return LLM, SamplingParams


def _int_config_attr(config: Any, name: str) -> int | None:
    value = getattr(config, name, None)
    if isinstance(value, int) and value > 0:
        return value
    return None


def _derive_num_attention_heads(config: Any) -> int | None:
    for name in ("num_attention_heads", "n_head", "num_heads", "n_heads"):
        value = _int_config_attr(config, name)
        if value is not None:
            return value

    hidden_size = _int_config_attr(config, "hidden_size") or _int_config_attr(config, "n_embd")
    head_dim = _int_config_attr(config, "head_dim") or _int_config_attr(config, "attention_head_size")
    if hidden_size is not None and head_dim is not None and hidden_size % head_dim == 0:
        return hidden_size // head_dim

    return _int_config_attr(config, "num_key_value_heads")


def _vllm_hf_overrides(model_id: str, *, local_files_only: bool) -> dict[str, int]:
    hf_config = AutoConfig.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    overrides: dict[str, int] = {}
    if not hasattr(hf_config, "num_attention_heads"):
        num_attention_heads = _derive_num_attention_heads(hf_config)
        if num_attention_heads is not None:
            overrides["num_attention_heads"] = num_attention_heads
    return overrides


def _signature_accepts(parameters: Iterable[inspect.Parameter], name: str) -> bool:
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters) or any(
        param.name == name for param in parameters
    )


def _trim_generated_ids(
    input_ids: list[int],
    *,
    prompt_length: int,
    eos_token_ids: TypingSequence[int],
    max_tokens: int | None,
) -> list[int]:
    if max_tokens is not None:
        input_ids = input_ids[:max_tokens]
    prompt_length = min(prompt_length, len(input_ids))
    eos_ids = set(eos_token_ids)
    for offset, token_id in enumerate(input_ids[prompt_length:]):
        if token_id in eos_ids:
            return input_ids[: prompt_length + offset + 1]
    return input_ids


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
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    timings = TimingStats()
    storage_dtype = _storage_torch_dtype(config.storage_dtype)
    print(f"loading tokenizer: {config.model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        local_files_only=config.local_files_only,
        trust_remote_code=True,
    )
    pad_token_id = _ensure_padding(tokenizer, getattr(tokenizer, "eos_token_id", None))
    eos_token_ids = _eos_token_ids(tokenizer, getattr(tokenizer, "eos_token_id", None))
    print(f"tokenizer loaded: {config.model_id}", flush=True)

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

    answer_rows: list[dict[str, Any]] = []
    with timed_section(timings, "teacher_collection"):
        total = min(config.limit, len(raw_dataset)) if config.limit is not None else len(raw_dataset)
        batches = list(_batched(_iter_limited(raw_dataset, config.limit), config.batch_size))
        print(f"loading vLLM generation model: {config.model_id}", flush=True)
        LLM, SamplingParams = _load_vllm()
        llm_signature = inspect.signature(LLM).parameters
        llm_kwargs: dict[str, Any] = {
            "model": config.model_id,
            "tokenizer": config.model_id,
            "dtype": config.dtype or "auto",
            "trust_remote_code": True,
        }
        hf_overrides = _vllm_hf_overrides(config.model_id, local_files_only=config.local_files_only)
        if hf_overrides and _signature_accepts(llm_signature.values(), "hf_overrides"):
            print(f"using vLLM HF config overrides: {hf_overrides}", flush=True)
            llm_kwargs["hf_overrides"] = hf_overrides
        elif hf_overrides:
            print(
                f"vLLM HF config overrides unavailable in this vLLM version: {hf_overrides}",
                flush=True,
            )
        if config.vllm_max_model_len is not None and _signature_accepts(llm_signature.values(), "max_model_len"):
            llm_kwargs["max_model_len"] = config.vllm_max_model_len
        if _signature_accepts(llm_signature.values(), "seed"):
            llm_kwargs["seed"] = config.seed
        with timed_section(timings, "generation_model_load"):
            llm = LLM(**llm_kwargs)
        print(f"vLLM generation model loaded: {config.model_id}", flush=True)
        generation_bar = tqdm(total=total, desc="phase 1/2 generating answers")
        with timed_section(timings, "teacher_generation"):
            sampling_signature = inspect.signature(SamplingParams).parameters
            sampling_kwargs: dict[str, Any] = {
                "temperature": 0.0,
                "max_tokens": config.generation_max_new_tokens,
                "stop_token_ids": eos_token_ids,
            }
            if "skip_special_tokens" in sampling_signature:
                sampling_kwargs["skip_special_tokens"] = False
            if "seed" in sampling_signature:
                sampling_kwargs["seed"] = config.seed
            sampling_params = SamplingParams(**sampling_kwargs)
            for batch in batches:
                batch_indices = [index for index, _ in batch]
                batch_examples = [example for _, example in batch]
                prompt_texts = [format_gsm8k_prompt(example, tokenizer) for example in batch_examples]
                request_outputs = llm.generate(prompt_texts, sampling_params)
                generation_bar.update(len(batch))
                for row_idx, output in enumerate(request_outputs):
                    prompt_ids = getattr(output, "prompt_token_ids", None)
                    if prompt_ids is None:
                        prompt_ids = tokenizer.encode(prompt_texts[row_idx], add_special_tokens=False)
                    generated_token_ids = output.outputs[0].token_ids
                    input_ids = list(prompt_ids) + list(generated_token_ids)
                    prompt_length = len(prompt_ids)
                    input_ids = _trim_generated_ids(
                        input_ids,
                        prompt_length=prompt_length,
                        eos_token_ids=eos_token_ids,
                        max_tokens=config.max_tokens,
                    )
                    if len(input_ids) < 2:
                        continue
                    prompt_length = min(prompt_length, len(input_ids))
                    generated_text = tokenizer.decode(input_ids[prompt_length:], skip_special_tokens=False)
                    text = prompt_texts[row_idx] + generated_text
                    answer_rows.append(
                        {
                            "text": text,
                            "prompt_text": prompt_texts[row_idx],
                            "generated_text": generated_text,
                            "input_ids": [int(token_id) for token_id in input_ids],
                            "example_id": str(batch_examples[row_idx].get("id", batch_indices[row_idx])),
                            "source": config.source,
                            "split": config.split,
                            "format_name": config.format_name,
                            "model_id": config.model_id,
                            "hidden_dtype": config.storage_dtype,
                            "num_tokens": int(len(input_ids)),
                            "prompt_length": int(prompt_length),
                        }
                    )
        generation_bar.close()
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with timed_section(timings, "tmp_answer_dataset_build"):
            answer_dataset = Dataset.from_list(answer_rows, features=teacher_answer_dataset_features())
        if config.tmp_output_dir:
            print(f"saving temporary answer dataset: {config.tmp_output_dir}", flush=True)
            answer_dataset.save_to_disk(config.tmp_output_dir)
        if config.tmp_push_to_hub:
            print(f"pushing temporary answer dataset: {config.tmp_push_to_hub}", flush=True)
            answer_dataset.push_to_hub(config.tmp_push_to_hub, private=config.private)

        print(f"loading hidden extraction model: {config.model_id}", flush=True)
        context = ChainedFlowContext.from_pretrained(
            config.model_id,
            device=config.device,
            dtype=_model_torch_dtype(config.dtype),
            local_files_only=config.local_files_only,
        )
        timings.merge(context.timings)
        wrapper = context.frozen_lm
        pad_token_id = _ensure_padding(wrapper.tokenizer, wrapper.eos_token_id)
        print(f"hidden extraction model loaded: {config.model_id}", flush=True)
        print(f"model device: {wrapper.device}", flush=True)

        rows: list[dict[str, Any]] = []
        hidden_bar = tqdm(total=len(answer_rows), desc="phase 2/2 extracting hidden states")
        with timed_section(timings, "teacher_hidden_extraction", wrapper.device):
            for batch in _batched(list(enumerate(answer_rows)), config.batch_size):
                batch_rows = [row for _, row in batch]
                sequences = [
                    torch.tensor(row["input_ids"], dtype=torch.long, device=wrapper.device)
                    for row in batch_rows
                ]
                padded_ids, attention_mask = _left_pad_sequences(
                    sequences,
                    pad_token_id=pad_token_id,
                    device=wrapper.device,
                )
                with timed_section(timings, "hidden_extraction", wrapper.device):
                    outputs = _backbone(wrapper.model)(
                        input_ids=padded_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                final_hidden = outputs.hidden_states[-1]
                for row_idx, row in enumerate(batch_rows):
                    length = int(row["num_tokens"])
                    left_pad = padded_ids.shape[1] - length
                    hidden = final_hidden[row_idx, left_pad : left_pad + length]
                    rows.append(
                        {
                            **row,
                            "final_hidden": hidden.detach().cpu().to(storage_dtype).tolist(),
                        }
                    )
                hidden_bar.update(len(batch_rows))
        hidden_bar.close()

    with timed_section(timings, "dataset_build"):
        dataset = Dataset.from_list(rows, features=teacher_dataset_features(config.storage_dtype))
    return dataset, timings

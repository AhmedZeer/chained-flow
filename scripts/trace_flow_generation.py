from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import codecs

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chained_flow.context import ChainedFlowContext
from chained_flow.frozen_lm import DEFAULT_MODEL_ID
from chained_flow.training.eval_chunked_flow import load_flow_training_module, torch_dtype_from_string
from chained_flow.training.train_chunked_flow import ChunkedFlowModelArguments
from chained_flow.training.eval_chunked_flow import find_flow_config_dir
from chained_flow.verifier import SpeculativeVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace greedy backbone generation, drafter-only rollout, and drafter+verifier rollout "
            "from the same prompt."
        )
    )
    parser.add_argument("--flow_dir", required=True, help="Flow checkpoint directory, e.g. .../checkpoint-98316.")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="Prompt text.")
    prompt.add_argument("--prompt_file", help="Path to a UTF-8 prompt text file.")
    parser.add_argument(
        "--decode_prompt_escapes",
        action="store_true",
        help=r"Decode escapes in --prompt, so '\n' becomes a real newline.",
    )
    parser.add_argument(
        "--chat_template",
        action="store_true",
        help="Treat the prompt as a user message and apply the tokenizer chat template.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=32, help="N: number of new tokens to trace.")
    parser.add_argument("--draft_len", type=int, default=None, help="K: drafted tokens per pass. Defaults to checkpoint K.")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", default=None)
    return parser.parse_args()


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    if args.decode_prompt_escapes:
        prompt = codecs.decode(prompt, "unicode_escape")
    return prompt


def render_prompt(frozen_lm, prompt: str, *, chat_template: bool) -> str:
    if not chat_template:
        return prompt
    tokenizer = frozen_lm.tokenizer
    messages = [{"role": "user", "content": prompt.strip()}]
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
    return prompt


def load_model_args(flow_dir: str | Path) -> ChunkedFlowModelArguments:
    config_dir = find_flow_config_dir(flow_dir)
    with (config_dir / "chained_flow_chunked_flow_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    return ChunkedFlowModelArguments(**config["model_args"])


def ids_list(tensor: torch.Tensor) -> list[int]:
    return [int(x) for x in tensor.detach().cpu().reshape(-1).tolist()]


def decode_ids(frozen_lm, token_ids: list[int]) -> str:
    return frozen_lm.decode(token_ids, skip_special_tokens=False)


def prefix_match_len(left: list[int], right: list[int]) -> int:
    total = 0
    for a, b in zip(left, right):
        if a != b:
            break
        total += 1
    return total


def seed_torch(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def trace_backbone(frozen_lm, input_ids: torch.Tensor, *, max_new_tokens: int) -> dict[str, Any]:
    state, prefill_timings = frozen_lm.prefill(input_ids)
    generated: list[int] = []
    steps: list[dict[str, Any]] = []

    for step_idx in range(max_new_tokens):
        prefix_len_before = int(state.position)
        token, next_timings = frozen_lm.next_token(state)
        state, forward_timings = frozen_lm.forward_with_cache(token, state, use_cache=True)
        token_id = int(token.item())
        generated.append(token_id)
        steps.append(
            {
                "step": step_idx + 1,
                "prefix_len_before": prefix_len_before,
                "token": token_id,
                "text": decode_ids(frozen_lm, [token_id]),
                "timings": {
                    "next_token": next_timings.get("next_token"),
                    "forward_with_cache": forward_timings.get("forward_with_cache"),
                },
            }
        )

    return {
        "generated_ids": generated,
        "generated_text": decode_ids(frozen_lm, generated),
        "full_ids": ids_list(state.input_ids),
        "full_text": frozen_lm.decode(state.input_ids, skip_special_tokens=False),
        "prefill_seconds": prefill_timings.get("prefill"),
        "steps": steps,
    }


@torch.inference_mode()
def trace_drafter_only(
    frozen_lm,
    drafter,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    draft_len: int,
    seed: int,
) -> dict[str, Any]:
    seed_torch(seed, frozen_lm.device)
    state, prefill_timings = frozen_lm.prefill(input_ids)
    generated: list[int] = []
    steps: list[dict[str, Any]] = []
    pass_idx = 0

    while len(generated) < max_new_tokens:
        pass_idx += 1
        prefix_len_before = int(state.position)
        remaining = max_new_tokens - len(generated)
        proposal = drafter.propose(state, min(draft_len, remaining))
        proposal_ids = ids_list(proposal.tokens)
        if not proposal_ids:
            break

        proposal_tensor = torch.tensor([proposal_ids], dtype=torch.long, device=frozen_lm.device)
        state, forward_timings = frozen_lm.forward_with_cache(proposal_tensor, state, use_cache=True)
        generated.extend(proposal_ids)

        steps.append(
            {
                "pass": pass_idx,
                "prefix_len_before": prefix_len_before,
                "draft_len": len(proposal_ids),
                "proposal_ids": proposal_ids,
                "proposal_text": decode_ids(frozen_lm, proposal_ids),
                "committed_ids": proposal_ids,
                "committed_text": decode_ids(frozen_lm, proposal_ids),
                "prefix_len_after": int(state.position),
                "timings": {
                    "drafter": proposal.timings.get("drafter_single_expert_flow"),
                    "forward_with_cache": forward_timings.get("forward_with_cache"),
                },
            }
        )

    generated = generated[:max_new_tokens]
    return {
        "generated_ids": generated,
        "generated_text": decode_ids(frozen_lm, generated),
        "full_ids": ids_list(state.input_ids[:, : input_ids.shape[1] + max_new_tokens]),
        "full_text": frozen_lm.decode(state.input_ids[:, : input_ids.shape[1] + max_new_tokens], skip_special_tokens=False),
        "prefill_seconds": prefill_timings.get("prefill"),
        "steps": steps,
    }


@torch.inference_mode()
def trace_drafter_with_verifier(
    frozen_lm,
    drafter,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    draft_len: int,
    seed: int,
) -> dict[str, Any]:
    seed_torch(seed, frozen_lm.device)
    state, prefill_timings = frozen_lm.prefill(input_ids)
    verifier = SpeculativeVerifier(frozen_lm)
    generated: list[int] = []
    steps: list[dict[str, Any]] = []
    pass_idx = 0
    fallback_count = 0
    accepted_total = 0

    while len(generated) < max_new_tokens:
        pass_idx += 1
        prefix_len_before = int(state.position)
        remaining = max_new_tokens - len(generated)
        clean_state, _ = frozen_lm.prefill(state.input_ids)
        clean_next_token, _ = frozen_lm.next_token(clean_state)
        clean_next_id = int(clean_next_token.item())
        proposal = drafter.propose(state, min(draft_len, remaining))
        proposal_ids = ids_list(proposal.tokens)
        if not proposal_ids:
            break

        verify_result = verifier.verify(state, proposal.tokens)
        accepted_len = min(int(verify_result.acceptance.accepted_len), remaining)
        accepted_ids = proposal_ids[:accepted_len]
        state = verify_result.state
        generated.extend(accepted_ids)
        accepted_total += accepted_len

        fallback_id: int | None = None
        fallback_text: str | None = None
        appended_fallback = False
        if accepted_len < len(proposal_ids) and len(generated) < max_new_tokens:
            fallback_token = verify_result.acceptance.next_token.to(frozen_lm.device)
            fallback_id = int(fallback_token.item())
            fallback_text = decode_ids(frozen_lm, [fallback_id])
            state, _ = frozen_lm.forward_with_cache(fallback_token, state, use_cache=True)
            generated.append(fallback_id)
            fallback_count += 1
            appended_fallback = True

        committed_ids = accepted_ids + ([fallback_id] if appended_fallback and fallback_id is not None else [])
        steps.append(
            {
                "pass": pass_idx,
                "prefix_len_before": prefix_len_before,
                "draft_len": len(proposal_ids),
                "proposal_ids": proposal_ids,
                "proposal_text": decode_ids(frozen_lm, proposal_ids),
                "clean_next_id": clean_next_id,
                "clean_next_text": decode_ids(frozen_lm, [clean_next_id]),
                "matches": ids_list(verify_result.acceptance.matches.long()),
                "accepted_len": accepted_len,
                "accepted_ids": accepted_ids,
                "accepted_text": decode_ids(frozen_lm, accepted_ids),
                "fallback_id": fallback_id,
                "fallback_text": fallback_text,
                "fallback_matches_clean_next": fallback_id == clean_next_id if fallback_id is not None else None,
                "committed_ids": committed_ids,
                "committed_text": decode_ids(frozen_lm, committed_ids),
                "prefix_len_after": int(state.position),
                "timings": {
                    "drafter": proposal.timings.get("drafter_single_expert_flow"),
                    "verifier_forward": verify_result.timings.get("verifier_forward"),
                    "acceptance": verify_result.timings.get("acceptance"),
                    "cache_repair": verify_result.timings.get("cache_repair"),
                },
            }
        )

    generated = generated[:max_new_tokens]
    mean_accept_len = accepted_total / len(steps) if steps else 0.0
    return {
        "generated_ids": generated,
        "generated_text": decode_ids(frozen_lm, generated),
        "full_ids": ids_list(state.input_ids[:, : input_ids.shape[1] + max_new_tokens]),
        "full_text": frozen_lm.decode(state.input_ids[:, : input_ids.shape[1] + max_new_tokens], skip_special_tokens=False),
        "prefill_seconds": prefill_timings.get("prefill"),
        "passes": len(steps),
        "accepted_total": accepted_total,
        "fallback_count": fallback_count,
        "mean_accept_len": mean_accept_len,
        "steps": steps,
    }


def print_step_trace(title: str, result: dict[str, Any]) -> None:
    print("", flush=True)
    print(title, flush=True)
    print("=" * len(title), flush=True)
    print(f"generated ids : {result['generated_ids']}", flush=True)
    print(f"generated text: {result['generated_text']!r}", flush=True)
    if "passes" in result:
        print(
            f"passes={result['passes']} accepted_total={result['accepted_total']} "
            f"fallback_count={result['fallback_count']} mean_accept_len={result['mean_accept_len']:.3f}",
            flush=True,
        )
    for step in result["steps"]:
        if "proposal_ids" not in step:
            print(
                f"step {step['step']:02d}: token={step['token']} text={step['text']!r} "
                f"prefix_before={step['prefix_len_before']}",
                flush=True,
            )
            continue
        if "accepted_len" not in step:
            print(
                f"pass {step['pass']:02d}: draft={step['proposal_ids']} "
                f"text={step['proposal_text']!r} prefix {step['prefix_len_before']}->{step['prefix_len_after']}",
                flush=True,
            )
            continue
        print(
            f"pass {step['pass']:02d}: draft={step['proposal_ids']} matches={step['matches']} "
            f"accepted={step['accepted_len']} fallback={step['fallback_id']} "
            f"clean_next={step['clean_next_id']} "
            f"commit={step['committed_ids']} text={step['committed_text']!r} "
            f"prefix {step['prefix_len_before']}->{step['prefix_len_after']}",
            flush=True,
        )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch_dtype_from_string(args.dtype)
    raw_prompt = read_prompt(args)
    model_args = load_model_args(args.flow_dir)
    draft_len = model_args.draft_length if args.draft_len is None else args.draft_len
    if draft_len < 1:
        raise ValueError("draft_len must be >= 1")

    print(f"loading backbone: model_id={args.model_id} dtype={args.dtype} device={device}", flush=True)
    context = ChainedFlowContext.from_pretrained(
        args.model_id,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    frozen_lm = context.frozen_lm

    print(f"loading flow checkpoint: {args.flow_dir}", flush=True)
    module, config = load_flow_training_module(args.flow_dir, frozen_lm=frozen_lm, device=device)

    prompt = render_prompt(frozen_lm, raw_prompt, chat_template=args.chat_template)
    input_ids = frozen_lm.tokenize(prompt)
    print("", flush=True)
    print("trace setup", flush=True)
    print("===========", flush=True)
    print(f"prompt tokens={input_ids.shape[1]} max_new_tokens={args.max_new_tokens} draft_len={draft_len}", flush=True)
    print(f"raw prompt text: {raw_prompt!r}", flush=True)
    print(f"rendered prompt text: {prompt!r}", flush=True)

    backbone = trace_backbone(frozen_lm, input_ids, max_new_tokens=args.max_new_tokens)
    drafter_only = trace_drafter_only(
        frozen_lm,
        module.drafter,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        draft_len=draft_len,
        seed=args.seed,
    )
    drafter_verifier = trace_drafter_with_verifier(
        frozen_lm,
        module.drafter,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        draft_len=draft_len,
        seed=args.seed,
    )

    summary = {
        "backbone_tokens": len(backbone["generated_ids"]),
        "drafter_only_tokens": len(drafter_only["generated_ids"]),
        "drafter_verifier_tokens": len(drafter_verifier["generated_ids"]),
        "drafter_only_prefix_match_vs_backbone": prefix_match_len(
            drafter_only["generated_ids"], backbone["generated_ids"]
        ),
        "drafter_verifier_prefix_match_vs_backbone": prefix_match_len(
            drafter_verifier["generated_ids"], backbone["generated_ids"]
        ),
        "drafter_verifier_exact_match_backbone": drafter_verifier["generated_ids"] == backbone["generated_ids"],
        "drafter_verifier_passes": drafter_verifier["passes"],
        "drafter_verifier_mean_accept_len": drafter_verifier["mean_accept_len"],
        "drafter_verifier_fallback_count": drafter_verifier["fallback_count"],
    }

    print_step_trace("1. Greedy Backbone", backbone)
    print_step_trace("2. Drafter Only", drafter_only)
    print_step_trace("3. Drafter + Verifier", drafter_verifier)

    print("", flush=True)
    print("summary", flush=True)
    print("=======", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)

    result = {
        "metadata": {
            "flow_dir": args.flow_dir,
            "model_id": args.model_id,
            "device": str(device),
            "dtype": args.dtype,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "draft_len": draft_len,
            "decode_prompt_escapes": args.decode_prompt_escapes,
            "chat_template": args.chat_template,
            "raw_prompt": raw_prompt,
            "rendered_prompt": prompt,
        },
        "config": config,
        "summary": summary,
        "backbone": backbone,
        "drafter_only": drafter_only,
        "drafter_verifier": drafter_verifier,
    }

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"trace saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()

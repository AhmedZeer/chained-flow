from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chained_flow.frozen_lm import DEFAULT_MODEL_ID
from chained_flow.training.collect_teacher import TeacherCollectionConfig, collect_teacher_dataset


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect K-independent frozen-LM teacher hidden states.")
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--dataset-name", default=os.getenv("DATASET_NAME", "gsm8k"))
    parser.add_argument("--dataset-config", default=os.getenv("DATASET_CONFIG", "main"))
    parser.add_argument("--split", default=os.getenv("DATASET_SPLIT", "train"))
    parser.add_argument("--source", default=os.getenv("SOURCE", "gsm8k"))
    parser.add_argument("--format-name", default=os.getenv("FORMAT_NAME", "qwen_chat_qa"))
    parser.add_argument("--limit", type=int, default=env_int("LIMIT"))
    parser.add_argument("--max-tokens", type=int, default=env_int("MAX_TOKENS"))
    parser.add_argument("--generation-max-new-tokens", type=int, default=env_int("GENERATION_MAX_NEW_TOKENS") or 4096)
    parser.add_argument(
        "--storage-dtype",
        choices=["float32", "float16", "bfloat16"],
        default=os.getenv("STORAGE_DTYPE", "float32"),
    )
    parser.add_argument("--local-files-only", action="store_true", default=env_bool("LOCAL_FILES_ONLY"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("OUTPUT_DIR", "teacher_states/gsm8k-qwen35-08b-smoke")),
    )
    parser.add_argument("--push-to-hub", default=os.getenv("PUSH_TO_HUB"), help="Optional HF repo id, e.g. user/dataset-name.")
    parser.add_argument("--private", action="store_true", default=env_bool("PRIVATE"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TeacherCollectionConfig(
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        source=args.source,
        format_name=args.format_name,
        limit=args.limit,
        max_tokens=args.max_tokens,
        generation_max_new_tokens=args.generation_max_new_tokens,
        storage_dtype=args.storage_dtype,
        local_files_only=args.local_files_only,
    )
    dataset, timings = collect_teacher_dataset(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))
    if args.push_to_hub:
        dataset.push_to_hub(args.push_to_hub, private=args.private)

    print(f"saved={args.output_dir}")
    print(f"rows={len(dataset)}")
    print(f"columns={dataset.column_names}")
    print(f"timings={timings.scalar_components() if hasattr(timings, 'scalar_components') else timings.sections}")


if __name__ == "__main__":
    main()

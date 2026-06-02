from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv
import yaml

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chained_flow.frozen_lm import DEFAULT_MODEL_ID
from chained_flow.training.collect_teacher import TeacherCollectionConfig, collect_teacher_dataset


def load_yaml_args(path: Path) -> argparse.Namespace:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("collection YAML config must contain a mapping at the top level")
    data["output_dir"] = Path(data.get("output_dir", "teacher_states/gsm8k-qwen35-08b-smoke"))
    return argparse.Namespace(
        model_id=data.get("model_id", DEFAULT_MODEL_ID),
        dataset_name=data.get("dataset_name", "gsm8k"),
        dataset_config=data.get("dataset_config", "main"),
        split=data.get("split", "train"),
        source=data.get("source", "gsm8k"),
        format_name=data.get("format_name", "qwen_chat_qa"),
        limit=data.get("limit"),
        max_tokens=data.get("max_tokens"),
        generation_max_new_tokens=data.get("generation_max_new_tokens", 4096),
        storage_dtype=data.get("storage_dtype", "float32"),
        local_files_only=bool(data.get("local_files_only", False)),
        output_dir=data["output_dir"],
        push_to_hub=data.get("push_to_hub"),
        private=bool(data.get("private", False)),
    )


def parse_args() -> argparse.Namespace:
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        return load_yaml_args(Path(sys.argv[1]))

    parser = argparse.ArgumentParser(description="Collect K-independent frozen-LM teacher hidden states.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset-name", default="gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--source", default="gsm8k")
    parser.add_argument("--format-name", default="qwen_chat_qa")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--generation-max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--storage-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("teacher_states/gsm8k-qwen35-08b-smoke"),
    )
    parser.add_argument("--push-to-hub", default=None, help="Optional HF repo id, e.g. user/dataset-name.")
    parser.add_argument("--private", action="store_true")
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

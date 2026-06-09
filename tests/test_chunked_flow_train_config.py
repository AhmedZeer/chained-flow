from pathlib import Path

from transformers import HfArgumentParser, TrainingArguments

from chained_flow.training.train_chunked_flow import (
    ChunkedFlowModelArguments,
    FlowLossArguments,
    TeacherDataArguments,
)


def test_chunked_flow_args_parse_minimal_cli():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )

    model_args, data_args, loss_args, training_args = parser.parse_args_into_dataclasses(
        [
            "--vae_dir",
            "out/vae/ckpts/hidden-vae-lr3e3/checkpoint-1880",
            "--dataset_path",
            "teacher_states/gsm8k-qwen35-08b-smoke",
            "--output_dir",
            "out/flow/ckpts/smoke",
            "--per_device_train_batch_size",
            "2",
        ]
    )

    assert model_args.draft_length == 2
    assert model_args.chunk_size == 2
    assert model_args.vae_dir == "out/vae/ckpts/hidden-vae-lr3e3/checkpoint-1880"
    assert data_args.dataset_path == "teacher_states/gsm8k-qwen35-08b-smoke"
    assert loss_args.gamma == 0.8
    assert training_args.output_dir == "out/flow/ckpts/smoke"


def test_chunked_flow_args_parse_yaml_config():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )

    model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
        yaml_file=str(Path("train_configs/chunked_flow/smoke_chunked_flow.yaml").resolve())
    )

    assert model_args.draft_length == 2
    assert model_args.chunk_size == 2
    assert model_args.vae_dir == "out/vae/ckpts/hidden-vae-lr3e3/checkpoint-1880"
    assert model_args.expert_dim == 64
    assert data_args.dataset_path == "teacher_states/gsm8k-qwen35-08b-smoke"
    assert data_args.dataset_split == "train"
    assert data_args.windows_per_epoch == 32
    assert data_args.materialize_rows is True
    assert loss_args.gamma == 0.8
    assert training_args.output_dir == "out/flow/ckpts/smoke-chunked-flow-k2"

def test_chunked_flow_sweep_yaml_configs_parse_and_use_unique_outputs():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )
    config_paths = sorted(Path("train_configs/chunked_flow/sweeps").glob("*/*.yaml"))

    assert len(config_paths) == 15

    output_dirs = []
    for config_path in config_paths:
        model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
            yaml_file=str(config_path.resolve())
        )
        assert model_args.chunk_size == model_args.draft_length
        assert data_args.dataset_path == "data/flow_cache/gsm8k_6.5k_train"
        assert data_args.materialize_rows is True
        assert loss_args.gamma == 0.8
        output_dirs.append(training_args.output_dir)

    assert len(output_dirs) == len(set(output_dirs))


def test_chunked_flow_k_sweep_yaml_configs_use_expected_grid():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )
    config_paths = sorted(Path("train_configs/chunked_flow/sweeps/k_draft").glob("*.yaml"))

    assert len(config_paths) == 6

    grid = set()
    for config_path in config_paths:
        model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
            yaml_file=str(config_path.resolve())
        )
        assert model_args.context_size == 8
        assert model_args.chunk_size == model_args.draft_length
        assert model_args.draft_length in {4, 6, 8}
        assert model_args.expert_dim in {384, 512}
        assert data_args.dataset_path == "data/flow_cache/gsm8k_6.5k_train"
        assert training_args.per_device_train_batch_size == 4096
        grid.add((model_args.draft_length, model_args.expert_dim))

    assert grid == {(4, 384), (4, 512), (6, 384), (6, 512), (8, 384), (8, 512)}


from datasets import Dataset
import torch

from chained_flow.training import vae_dataset
from chained_flow.training.vae_dataset import TeacherHiddenTokenDataset, collate_hidden_tokens


def test_teacher_hidden_token_dataset_samples_response_tokens():
    dataset = Dataset.from_list(
        [
            {
                "final_hidden": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                "num_tokens": 3,
                "prompt_length": 1,
            }
        ]
    )
    token_dataset = TeacherHiddenTokenDataset(dataset, tokens_per_epoch=4, seed=0, response_only=True)
    item = token_dataset[0]

    assert item["hidden"].shape == (2,)
    assert item["hidden"][0].item() in {1.0, 2.0}


def test_teacher_hidden_token_dataset_init_uses_metadata_columns_only():
    dataset = Dataset.from_dict(
        {
            "final_hidden": [[[0.0, 0.0], [1.0, 1.0]]],
            "num_tokens": [2],
            "prompt_length": [1],
        }
    )
    dataset = dataset.remove_columns("final_hidden")
    token_dataset = TeacherHiddenTokenDataset(dataset, tokens_per_epoch=1, seed=0, response_only=True)

    assert token_dataset.valid_rows == [(0, 1, 2)]


def test_collate_hidden_tokens_stacks_batch():
    batch = [{"hidden": torch.zeros(2)}, {"hidden": torch.ones(2)}]
    collated = collate_hidden_tokens(batch)

    assert collated["hidden"].shape == (2, 2)


def test_teacher_hidden_token_dataset_loads_hf_dataset_when_path_is_not_local(monkeypatch):
    dataset = Dataset.from_list(
        [
            {
                "final_hidden": [[0.0, 0.0], [1.0, 1.0]],
                "num_tokens": 2,
                "prompt_length": 1,
            }
        ]
    )
    calls = {}

    def fake_load_dataset(path, *, split):
        calls["path"] = path
        calls["split"] = split
        return dataset

    monkeypatch.setattr(vae_dataset, "load_dataset", fake_load_dataset)
    token_dataset = TeacherHiddenTokenDataset.from_path("user/repo", split="train", tokens_per_epoch=1)

    assert calls == {"path": "user/repo", "split": "train"}
    assert len(token_dataset) == 1

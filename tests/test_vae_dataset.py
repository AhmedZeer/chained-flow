from datasets import Dataset
import torch

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


def test_collate_hidden_tokens_stacks_batch():
    batch = [{"hidden": torch.zeros(2)}, {"hidden": torch.ones(2)}]
    collated = collate_hidden_tokens(batch)

    assert collated["hidden"].shape == (2, 2)

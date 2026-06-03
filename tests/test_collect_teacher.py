import torch

from chained_flow.training.collect_teacher import (
    _backbone,
    _model_torch_dtype,
    _sequence_spans,
    format_gsm8k_prompt,
    teacher_dataset_features,
)


def test_teacher_features_text_first_and_no_attention_mask():
    features = teacher_dataset_features()
    assert list(features.keys())[0] == "text"
    assert list(features.keys())[1] == "prompt_text"
    assert list(features.keys())[2] == "generated_text"
    assert "attention_mask" not in features
    assert "prompt_length" in features


def test_gsm8k_format_uses_prompt_only_without_reference_answer(fake_wrapper):
    text = format_gsm8k_prompt(
        {"question": "What is 2+2?", "answer": "#### 4"},
        fake_wrapper.tokenizer,
    )
    assert "What is 2+2?" in text
    assert "#### 4" not in text


def test_sequence_spans_trim_left_padding_and_stop_at_eos():
    spans = _sequence_spans(
        torch.tensor(
            [
                [0, 0, 4, 5, 6],
                [0, 7, 8, 2, 9],
            ]
        ),
        pad_token_id=0,
        eos_token_id=2,
        prompt_lengths=[2, 2],
    )
    assert spans == [(2, 5), (1, 4)]


def test_sequence_spans_ignore_eos_inside_prompt():
    spans = _sequence_spans(
        torch.tensor([[0, 4, 2, 5, 6, 2, 9]]),
        pad_token_id=0,
        eos_token_id=2,
        prompt_lengths=[4],
    )
    assert spans == [(1, 6)]


def test_model_dtype_parser():
    assert _model_torch_dtype(None) is None
    assert _model_torch_dtype("bfloat16") is torch.bfloat16


def test_backbone_prefers_model_attr(fake_wrapper):
    class ModelWithBackbone:
        def __init__(self):
            self.model = object()

    model = ModelWithBackbone()
    assert _backbone(model) is model.model

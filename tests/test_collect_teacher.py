from chained_flow.training.collect_teacher import format_gsm8k_prompt, teacher_dataset_features


def test_teacher_features_text_first_and_no_attention_mask():
    features = teacher_dataset_features()
    assert list(features.keys())[0] == "text"
    assert "attention_mask" not in features


def test_gsm8k_format_uses_prompt_only_without_reference_answer(fake_wrapper):
    text = format_gsm8k_prompt(
        {"question": "What is 2+2?", "answer": "#### 4"},
        fake_wrapper.tokenizer,
    )
    assert "What is 2+2?" in text
    assert "#### 4" not in text

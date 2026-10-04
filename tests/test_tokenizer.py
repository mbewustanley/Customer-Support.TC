from pathlib import Path

import pytest
from datasets import Dataset

from src.tokenization.tokenizer import Banking77Tokenizer


@pytest.fixture
def sample_dataset():
    return Dataset.from_dict(
        {
            "text": [
                "I am still waiting for my card",
                "How do I reset my pin?",
                "I want to check my balance",
            ],
            "label": [11, 21, 5],
        }
    )


@pytest.fixture
def tokenizer(tmp_path):
    return Banking77Tokenizer(
        model_name="distilbert-base-uncased",
        processed_data_dir=tmp_path / "processed",
        tokenized_data_dir=tmp_path / "tokenized",
        max_length=16,
    )


def test_tokenizer_initializes(tokenizer):
    assert tokenizer.model_name == "distilbert-base-uncased"
    assert tokenizer.max_length == 16
    assert tokenizer.tokenizer is not None


def test_tokenize_dataset_creates_required_columns(
    tokenizer,
    sample_dataset,
):
    result = tokenizer.tokenize_dataset(sample_dataset)

    assert "input_ids" in result.column_names
    assert "attention_mask" in result.column_names
    assert "label" in result.column_names


def test_tokenize_dataset_preserves_labels(
    tokenizer,
    sample_dataset,
):
    result = tokenizer.tokenize_dataset(sample_dataset)

    assert result["label"] == sample_dataset["label"]


def test_tokenize_dataset_respects_max_length(
    tokenizer,
    sample_dataset,
):
    result = tokenizer.tokenize_dataset(sample_dataset)

    for input_ids in result["input_ids"]:
        assert len(input_ids) <= 16

    for attention_mask in result["attention_mask"]:
        assert len(attention_mask) <= 16


def test_tokenize_dataset_requires_text_column(tokenizer):
    invalid_dataset = Dataset.from_dict(
        {
            "message": ["hello"],
            "label": [0],
        }
    )

    with pytest.raises(ValueError, match="text"):
        tokenizer.tokenize_dataset(invalid_dataset)


def test_tokenize_dataset_requires_label_column(tokenizer):
    invalid_dataset = Dataset.from_dict(
        {
            "text": ["hello"],
        }
    )

    with pytest.raises(ValueError, match="label"):
        tokenizer.tokenize_dataset(invalid_dataset)


def test_save_datasets(
    tokenizer,
    sample_dataset,
):
    tokenized = tokenizer.tokenize_dataset(sample_dataset)

    tokenizer.save_dataset(
        tokenized,
        tokenized,
    )

    train_path = (
        tokenizer.tokenized_data_dir / "train"
    )
    test_path = (
        tokenizer.tokenized_data_dir / "test"
    )

    assert train_path.exists()
    assert test_path.exists()


def test_run_with_real_processed_data(
    tmp_path,
):
    processed_dir = Path("data/processed")

    if not (
        processed_dir / "train"
    ).exists():
        pytest.skip(
            "Processed Banking77 data is not available."
        )

    tokenizer = Banking77Tokenizer(
        model_name="distilbert-base-uncased",
        processed_data_dir=processed_dir,
        tokenized_data_dir=tmp_path / "tokenized",
        max_length=128,
    )

    train, test = tokenizer.run()

    assert len(train) == 10003
    assert len(test) == 3080

    assert "input_ids" in train.column_names
    assert "attention_mask" in train.column_names
    assert "label" in train.column_names
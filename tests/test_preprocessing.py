import pytest
from datasets import Dataset

from src.preprocessing.preprocessing import Banking77Preprocessor


def create_preprocessor(tmp_path):
    config_path = "configs/data_config.yaml"

    return Banking77Preprocessor(
        config_path=config_path,
        raw_data_dir=str(tmp_path / "raw"),
        processed_data_dir=str(tmp_path / "processed"),
    )


def create_dataset(texts=None, labels=None):
    if texts is None:
        texts = [
            "I need help with my card",
            "Where is my cash withdrawal",
            "I forgot my PIN",
        ]

    if labels is None:
        labels = [0, 1, 2]

    return Dataset.from_dict(
        {
            "text": texts,
            "label": labels,
        }
    )


def test_preprocessor_loads_config(tmp_path):
    preprocessor = create_preprocessor(tmp_path)

    assert preprocessor.config is not None
    assert preprocessor.num_classes == 77


def test_text_whitespace_is_cleaned(tmp_path):
    preprocessor = create_preprocessor(tmp_path)

    dataset = create_dataset(
        texts=[
            "  I need help with my card  ",
            "Where is my cash withdrawal   ",
            "   I forgot my PIN",
        ]
    )

    result = preprocessor.clean_text(dataset)

    assert result["text"] == [
        "I need help with my card",
        "Where is my cash withdrawal",
        "I forgot my PIN",
    ]


def test_empty_text_records_are_removed(tmp_path):
    preprocessor = create_preprocessor(tmp_path)

    dataset = create_dataset(
        texts=[
            "I need help",
            "",
            "   ",
            "Where is my card?",
        ],
        labels=[0, 1, 2, 3],
    )

    cleaned = preprocessor.clean_text(dataset)
    result = preprocessor.remove_invalid_records(cleaned)

    assert len(result) == 2
    assert result["text"] == [
        "I need help",
        "Where is my card?",
    ]


def test_none_text_records_are_removed(tmp_path):
    preprocessor = create_preprocessor(tmp_path)

    dataset = Dataset.from_dict(
        {
            "text": [
                "I need help",
                None,
                "Where is my card?",
            ],
            "label": [0, 1, 2],
        }
    )

    result = preprocessor.remove_invalid_records(dataset)

    assert len(result) == 2
    assert result["text"] == [
        "I need help",
        "Where is my card?",
    ]


def test_valid_labels_are_preserved(tmp_path):
    preprocessor = create_preprocessor(tmp_path)

    dataset = create_dataset(
        labels=[0, 25, 76]
    )

    result = preprocessor.normalize_labels(dataset)

    assert result["label"] == [0, 25, 76]


def test_labels_are_converted_to_integers(tmp_path):
    preprocessor = create_preprocessor(tmp_path)

    dataset = Dataset.from_dict(
        {
            "text": [
                "First example",
                "Second example",
            ],
            "label": [
                1.0,
                2.0,
            ],
        }
    )

    result = preprocessor.normalize_labels(dataset)

    assert result["label"] == [1, 2]


def test_negative_labels_are_rejected(tmp_path):
    preprocessor = create_preprocessor(tmp_path)

    dataset = create_dataset(
        labels=[0, -1, 2]
    )

    with pytest.raises(Exception):
        preprocessor.normalize_labels(dataset)


def test_labels_above_range_are_rejected(tmp_path):
    preprocessor = create_preprocessor(tmp_path)

    dataset = create_dataset(
        labels=[0, 1, 77]
    )

    with pytest.raises(Exception):
        preprocessor.normalize_labels(dataset)


def test_preprocessing_preserves_valid_records(tmp_path):
    preprocessor = create_preprocessor(tmp_path)

    dataset = create_dataset()

    result = preprocessor.preprocess_dataset(dataset)

    assert len(result) == len(dataset)
    assert result["text"] == dataset["text"]
    assert result["label"] == dataset["label"]


def test_preprocessing_removes_invalid_records(tmp_path):
    preprocessor = create_preprocessor(tmp_path)

    dataset = create_dataset(
        texts=[
            "  I need help  ",
            "",
            "   ",
            "Where is my card?",
        ],
        labels=[0, 1, 2, 3],
    )

    result = preprocessor.preprocess_dataset(dataset)

    assert len(result) == 2

    assert result["text"] == [
        "I need help",
        "Where is my card?",
    ]

    assert result["label"] == [0, 3]


def test_processed_data_can_be_saved(tmp_path):
    preprocessor = create_preprocessor(tmp_path)

    train_dataset = create_dataset(
        texts=[
            "I need help",
            "Where is my card?",
        ],
        labels=[0, 1],
    )

    test_dataset = create_dataset(
        texts=[
            "I forgot my PIN",
            "My card is missing",
        ],
        labels=[2, 3],
    )

    preprocessor.save_dataset(
        train_dataset,
        test_dataset,
    )

    assert (tmp_path / "processed" / "train").exists()
    assert (tmp_path / "processed" / "test").exists()
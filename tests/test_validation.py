# i obviously do not want to test the entire banking77 dataset everytime
# hence, this script is designed to test the validation of the 
# dataset ingestion process using a small sample of the HUGGING FACE BANKING77 dataset.

# run this file in the root directory of the project using the command:
# pytest tests/test_validation.py -v


import pytest

from datasets import Dataset

from src.data.validation import Banking77Validator


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def create_validator():
    """Create a validator instance for testing."""

    return Banking77Validator(
        config_path="configs/data_config.yaml",
        raw_data_dir="data/raw",
    )


def create_dataset(
    texts=None,
    labels=None,
):
    """Create a small Hugging Face dataset for testing."""

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


# ----------------------------------------------------------------------
# Basic validator tests
# ----------------------------------------------------------------------

def test_validator_loads_config():

    validator = create_validator()

    assert validator.config is not None
    assert validator.num_classes == 77
    assert validator.label_indexing == "zero_based"


# ----------------------------------------------------------------------
# Column validation
# ----------------------------------------------------------------------

def test_valid_columns():

    validator = create_validator()

    dataset = create_dataset()

    validator.validate_columns(
        dataset,
        "train",
    )

    assert len(validator.errors) == 0


def test_missing_column():

    validator = create_validator()

    dataset = Dataset.from_dict(
        {
            "text": [
                "I need help"
            ]
        }
    )

    validator.validate_columns(
        dataset,
        "train",
    )

    assert len(validator.errors) == 1
    assert "Missing columns" in validator.errors[0]


def test_unexpected_column():

    validator = create_validator()

    dataset = Dataset.from_dict(
        {
            "text": ["I need help"],
            "label": [0],
            "extra": ["unexpected"],
        }
    )

    validator.validate_columns(
        dataset,
        "train",
    )

    assert len(validator.errors) == 1
    assert "Unexpected columns" in validator.errors[0]


# ----------------------------------------------------------------------
# Missing values
# ----------------------------------------------------------------------

def test_missing_values_are_detected():

    validator = create_validator()

    dataset = Dataset.from_dict(
        {
            "text": [
                "I need help",
                None,
            ],
            "label": [
                0,
                1,
            ],
        }
    )

    validator.validate_missing_values(
        dataset,
        "train",
    )

    assert len(validator.errors) == 1
    assert "missing values" in validator.errors[0]


# ----------------------------------------------------------------------
# Text validation
# ----------------------------------------------------------------------

def test_empty_text_is_detected():

    validator = create_validator()

    dataset = create_dataset(
        texts=[
            "I need help",
            "",
            "Where is my card?",
        ],
        labels=[
            0,
            1,
            2,
        ],
    )

    validator.validate_text(
        dataset,
        "train",
    )

    assert any(
        "empty strings" in error
        for error in validator.errors
    )


def test_whitespace_text_is_detected():

    validator = create_validator()

    dataset = create_dataset(
        texts=[
            "I need help",
            "     ",
            "Where is my card?",
        ],
        labels=[
            0,
            1,
            2,
        ],
    )

    validator.validate_text(
        dataset,
        "train",
    )

    assert any(
        "whitespace-only" in error
        for error in validator.errors
    )


# ----------------------------------------------------------------------
# Duplicate validation
# ----------------------------------------------------------------------

def test_duplicate_rows_generate_warning():

    validator = create_validator()

    dataset = create_dataset(
        texts=[
            "I need help",
            "I need help",
        ],
        labels=[
            0,
            0,
        ],
    )

    validator.validate_duplicates(
        dataset,
        "train",
    )

    assert len(validator.warnings) > 0

    assert any(
        "duplicate rows" in warning
        for warning in validator.warnings
    )


def test_duplicate_text_generates_warning():

    validator = create_validator()

    dataset = create_dataset(
        texts=[
            "I need help",
            "I need help",
        ],
        labels=[
            0,
            1,
        ],
    )

    validator.validate_duplicates(
        dataset,
        "train",
    )

    assert any(
        "duplicate text" in warning
        for warning in validator.warnings
    )


# ----------------------------------------------------------------------
# Label validation
# ----------------------------------------------------------------------

def test_labels_within_valid_range():

    validator = create_validator()

    # We don't need all 77 classes for this isolated test.
    # We're testing that labels themselves are integers.
    dataset = create_dataset(
        labels=[
            0,
            1,
            2,
        ]
    )

    validator.validate_labels(
        dataset,
        "train",
    )

    # This WILL produce an error because only 3 of 77 classes
    # are present. The important part is that no invalid-range
    # error is produced.
    assert not any(
        "outside the valid range" in error
        for error in validator.errors
    )


def test_negative_label_is_detected():

    validator = create_validator()

    dataset = create_dataset(
        labels=[
            -1,
            1,
            2,
        ]
    )

    validator.validate_labels(
        dataset,
        "train",
    )

    assert any(
        "outside the valid range" in error
        for error in validator.errors
    )


def test_label_above_expected_range_is_detected():

    validator = create_validator()

    dataset = create_dataset(
        labels=[
            0,
            1,
            77,
        ]
    )

    validator.validate_labels(
        dataset,
        "train",
    )

    assert any(
        "outside the valid range" in error
        for error in validator.errors
    )


# ----------------------------------------------------------------------
# Class count
# ----------------------------------------------------------------------

def test_missing_classes_are_detected():

    validator = create_validator()

    dataset = create_dataset(
        labels=[
            0,
            1,
            2,
        ]
    )

    validator.validate_labels(
        dataset,
        "train",
    )

    assert any(
        "unique classes" in error
        for error in validator.errors
    )

    assert any(
        "Missing label classes" in error
        for error in validator.errors
    )


# ----------------------------------------------------------------------
# Train/test leakage
# ----------------------------------------------------------------------

def test_train_test_leakage_is_detected():

    validator = create_validator()

    validator.train_dataset = create_dataset(
        texts=[
            "I need help",
            "Where is my card?",
        ],
        labels=[
            0,
            1,
        ],
    )

    validator.test_dataset = create_dataset(
        texts=[
            "I need help",
            "Different sentence",
        ],
        labels=[
            0,
            2,
        ],
    )

    validator.validate_train_test_leakage()

    assert any(
        "leakage detected" in error
        for error in validator.errors
    )


def test_no_train_test_leakage():

    validator = create_validator()

    validator.train_dataset = create_dataset(
        texts=[
            "I need help",
        ],
        labels=[
            0,
        ],
    )

    validator.test_dataset = create_dataset(
        texts=[
            "Completely different sentence",
        ],
        labels=[
            1,
        ],
    )

    validator.validate_train_test_leakage()

    assert not any(
        "leakage detected" in error
        for error in validator.errors
    )
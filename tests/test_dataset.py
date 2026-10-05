import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from src.data.dataset import Banking77Dataset


@pytest.fixture
def tokenized_dataset():
    return Dataset.from_dict(
        {
            "input_ids": [
                [101, 2023, 2003, 1037, 2742, 102],
                [101, 2129, 2079, 1045, 2515, 102],
                [101, 1045, 2293, 2026, 2502, 102],
            ],
            "attention_mask": [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            "label": [0, 1, 2],
        }
    )


@pytest.fixture
def banking_dataset(tokenized_dataset):
    return Banking77Dataset(tokenized_dataset)


def test_dataset_length(
    banking_dataset,
):
    assert len(banking_dataset) == 3


def test_dataset_returns_dictionary(
    banking_dataset,
):
    sample = banking_dataset[0]

    assert isinstance(sample, dict)

    assert set(sample.keys()) == {
        "input_ids",
        "attention_mask",
        "labels",
    }


def test_dataset_returns_tensors(
    banking_dataset,
):
    sample = banking_dataset[0]

    assert isinstance(
        sample["input_ids"],
        torch.Tensor,
    )

    assert isinstance(
        sample["attention_mask"],
        torch.Tensor,
    )

    assert isinstance(
        sample["labels"],
        torch.Tensor,
    )


def test_dataset_tensor_dtypes(
    banking_dataset,
):
    sample = banking_dataset[0]

    assert sample["input_ids"].dtype == torch.long
    assert sample["attention_mask"].dtype == torch.long
    assert sample["labels"].dtype == torch.long


def test_dataset_preserves_values(
    banking_dataset,
):
    sample = banking_dataset[1]

    assert sample["input_ids"].tolist() == [
        101,
        2129,
        2079,
        1045,
        2515,
        102,
    ]

    assert sample["attention_mask"].tolist() == [
        1,
        1,
        1,
        1,
        1,
        1,
    ]

    assert sample["labels"].item() == 1


def test_dataloader_batch_shape(
    banking_dataset,
):
    loader = DataLoader(
        banking_dataset,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(loader))

    assert batch["input_ids"].shape == (2, 6)
    assert batch["attention_mask"].shape == (2, 6)
    assert batch["labels"].shape == (2,)


def test_dataloader_returns_expected_batches(
    banking_dataset,
):
    loader = DataLoader(
        banking_dataset,
        batch_size=2,
        shuffle=False,
    )

    batches = list(loader)

    assert len(batches) == 2

    assert batches[0]["labels"].tolist() == [0, 1]
    assert batches[1]["labels"].tolist() == [2]


def test_dataloader_shuffling(
    banking_dataset,
):
    loader = DataLoader(
        banking_dataset,
        batch_size=3,
        shuffle=False,
    )

    batch = next(iter(loader))

    assert batch["labels"].tolist() == [0, 1, 2]


def test_real_tokenized_data():
    from pathlib import Path
    from datasets import load_from_disk

    train_path = Path("data/tokenized/train")

    if not train_path.exists():
        pytest.skip(
            "Real tokenized Banking77 data is not available."
        )

    dataset = load_from_disk(
        str(train_path)
    )

    pytorch_dataset = Banking77Dataset(
        dataset
    )

    assert len(pytorch_dataset) == 10003

    sample = pytorch_dataset[0]

    assert sample["input_ids"].shape == (128,)
    assert sample["attention_mask"].shape == (128,)
    assert sample["labels"].ndim == 0

    loader = DataLoader(
        pytorch_dataset,
        batch_size=16,
        shuffle=True,
    )

    batch = next(iter(loader))

    assert batch["input_ids"].shape == (
        16,
        128,
    )

    assert batch["attention_mask"].shape == (
        16,
        128,
    )

    assert batch["labels"].shape == (16,)

    assert torch.all(
        (batch["labels"] >= 0)
        & (batch["labels"] < 77)
    )
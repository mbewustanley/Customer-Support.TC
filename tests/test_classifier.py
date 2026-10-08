import pytest
import torch

from src.models.classifier import Banking77Classifier


@pytest.fixture
def model():
    return Banking77Classifier(
        model_name="distilbert-base-uncased",
        num_labels=77
    )


@pytest.fixture
def sample_batch():
    batch_size = 2
    sequence_length = 128

    return {
        "input_ids": torch.randint(
            low=0,
            high=30522,
            size=(batch_size, sequence_length),
            dtype=torch.long,
        ),
        "attention_mask": torch.ones(
            batch_size,
            sequence_length,
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [0, 76],
            dtype=torch.long,
        ),
    }


def test_model_initializes(model):
    assert model.model is not None
    assert model.model.num_labels == 77


def test_model_name(model):
    assert model.model_name == "distilbert-base-uncased"


def test_model_num_labels(model):
    assert model.num_labels == 77


def test_forward_pass_returns_logits(
    model,
    sample_batch,
):
    outputs = model(
        input_ids=sample_batch["input_ids"],
        attention_mask=sample_batch["attention_mask"]
    )

    assert hasattr(outputs, "logits")

    assert outputs.logits.shape == (
        2,
        77,
    )


def test_forward_pass_returns_loss(
    model,
    sample_batch
):
    outputs = model(
        input_ids=sample_batch["input_ids"],
        attention_mask=sample_batch["attention_mask"],
        labels=sample_batch["labels"],
    )

    assert outputs.loss is not None
    assert outputs.loss.ndim == 0
    assert torch.isfinite(outputs.loss)


def test_model_accepts_complete_training_batch(
    model,
    sample_batch
):
    outputs = model(**sample_batch)

    assert outputs.logits.shape == (
        2,
        77,
    )

    assert outputs.loss is not None


def test_model_runs_on_cpu(
    model,
    sample_batch
):
    model.to("cpu")

    outputs = model(
        input_ids=sample_batch["input_ids"],
        attention_mask=sample_batch["attention_mask"],
        labels=sample_batch["labels"],
    )

    assert outputs.logits.device.type == "cpu"


@pytest.mark.parametrize(
    "num_labels",
    [0, -1]
)
def test_invalid_num_labels(num_labels):
    with pytest.raises(ValueError):
        Banking77Classifier(
            model_name="distilbert-base-uncased",
            num_labels=num_labels,
        )
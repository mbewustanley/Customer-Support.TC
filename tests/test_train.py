from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from src.training import train
from datasets import Dataset


def test_split_training_dataset():
    dataset = Dataset.from_dict(
        {
            "text": [f"text {i}" for i in range(100)],
            "label": [i % 10 for i in range(100)]
        }
    ) 

    train_dataset, validation_dataset = train.split_training_dataset(
        dataset=dataset,
        validation_ratio=0.1,
        seed=42
    )

    assert len(train_dataset) == 90
    assert len(validation_dataset) == 10

    train_texts = set(train_dataset["text"])
    validation_texts = set(validation_dataset["text"])

    assert train_texts.isdisjoint(validation_texts)
    assert len(train_texts | validation_texts) == 100



def test_split_training_dataset_is_reproducible():
    dataset = Dataset.from_dict(
        {
            "text": [f"text {i}" for i in range(100)],
            "label": [i % 10 for i in range(100)],
        }
    )

    train_1, val_1 = train.split_training_dataset(
        dataset, validation_ratio=0.1, seed=42
    )

    train_2, val_2 = train.split_training_dataset(
        dataset, validation_ratio=0.1, seed=42
    )

    assert train_1["text"] == train_2["text"]
    assert val_1["text"] == val_2["text"]



def test_split_training_dataset_rejects_invalid_ratio():
    dataset = Dataset.from_dict(
        {
            "text": ["a", "b"],
            "label": [0, 1],
        }
    )

    with pytest.raises(ValueError):
        train.split_training_dataset(
            dataset,
            validation_ratio=0,
            seed=42,
        )

    with pytest.raises(ValueError):
        train.split_training_dataset(
            dataset,
            validation_ratio=1,
            seed=42,
        )


def test_create_dataloaders(tmp_path):

    from datasets import Dataset

    train_data = Dataset.from_dict(
    {
        "input_ids": [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24],
            [25, 26, 27],
            [28, 29, 30],
            [31, 32, 33],
            [34, 35, 36],
            [37, 38, 39],
            [40, 41, 42],
            [43, 44, 45],
            [46, 47, 48],
            [49, 50, 51],
            [52, 53, 54],
            [55, 56, 57],
            [58, 59, 60],
        ],
        "attention_mask": [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        "label": [0] * 10 + [1] * 10,
    }
)

    test_data = Dataset.from_dict(
        {
            "input_ids": [
                [7, 8, 9],
            ],
            "attention_mask": [
                [1, 1, 1],
            ],
            "label": [0],
        }
    )

    tokenized_dir = tmp_path / "tokenized"

    train_data.save_to_disk(
        str(tokenized_dir / "train")
    )

    test_data.save_to_disk(
        str(tokenized_dir / "test")
    )

    config = {
        "training": {
            "tokenized_data_dir": str(
                tokenized_dir
            ),
            "batch_size": 2,
            "eval_batch": 2,
            "num_workers": 0,
            "validation_ratio": 0.1,
            "seed": 42
        }
    }

    train_loader, validation_loader, test_loader = (
        train.create_dataloaders(config)
    )

    assert isinstance(
        train_loader,
        DataLoader,
    )

    assert isinstance(
        validation_loader,
        DataLoader,
    )

    assert isinstance(
            test_loader,
            DataLoader,
        )

    assert len(train_loader.dataset) == 18
    assert len(validation_loader.dataset) == 2
    assert len(test_loader.dataset) == 1


def test_create_dataloaders_missing_train(
    tmp_path,
):

    config = {
        "training": {
            "tokenized_data_dir": str(
                tmp_path / "tokenized"
            ),
            "batch_size": 2,
            "eval_batch": 1,
            "num_workers": 0,
        }
    }

    with pytest.raises(FileNotFoundError):

        train.create_dataloaders(config)


def test_create_model():

    config = {
        "training": {
            "model_name": "distilbert-base-uncased",
            "num_labels": 3,
        }
    }

    model = train.create_model(config)

    assert model.model_name == (
        "distilbert-base-uncased"
    )

    assert model.num_labels == 3


def test_ensure_tokenized_data_skips_existing(
    tmp_path,
    monkeypatch,
):

    tokenized_dir = tmp_path / "tokenized"

    (tokenized_dir / "train").mkdir(
        parents=True
    )

    (tokenized_dir / "test").mkdir()

    config = {
        "training": {
            "model_name": "distilbert-base-uncased",
            "tokenized_data_dir": str(
                tokenized_dir
            ),
            "max_length": 128,
        }
    }

    tokenizer_called = False

    def fake_tokenizer(*args, **kwargs):

        nonlocal tokenizer_called
        tokenizer_called = True

        raise AssertionError(
            "Tokenizer should not be called."
        )

    monkeypatch.setattr(
        train,
        "Banking77Tokenizer",
        fake_tokenizer,
    )

    train.ensure_tokenized_data(config)

    assert tokenizer_called is False


def test_ensure_tokenized_data_runs_tokenizer(
    tmp_path,
    monkeypatch,
):

    tokenized_dir = tmp_path / "tokenized"

    config = {
        "training": {
            "model_name": "distilbert-base-uncased",
            "tokenized_data_dir": str(
                tokenized_dir
            ),
            "max_length": 128,
        }
    }

    calls = {}

    class FakeTokenizer:

        def __init__(self, **kwargs):

            calls["init"] = kwargs

        def run(self):

            calls["run"] = True

    monkeypatch.setattr(
        train,
        "Banking77Tokenizer",
        FakeTokenizer,
    )

    train.ensure_tokenized_data(config)

    assert "init" in calls
    assert calls["init"]["model_name"] == (
        "distilbert-base-uncased"
    )
    assert calls["init"]["max_length"] == 128
    assert calls["run"] is True


def test_run_training(
    tmp_path,
    monkeypatch,
):

    config_path = tmp_path / "training.yaml"

    config_path.write_text(
        "dummy: config"
    )

    config = {
        "training": {
            "model_name": "test-model",
            "num_labels": 77,
            "tokenized_data_dir": (
                "data/tokenized"
            ),
            "report_dir": str(
                tmp_path / "reports"
            ),
        },
        "evaluation": {
            "metrics": [
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
            ]
        },
    }

    monkeypatch.setattr(
        train,
        "load_config",
        lambda path: config,
    )

    monkeypatch.setattr(
        train,
        "ensure_tokenized_data",
        lambda config: None,
    )

    fake_train_loader = object()
    fake_validation_loader = object()
    fake_test_loader = object()

    monkeypatch.setattr(
        train,
        "create_dataloaders",
        lambda config: (
            fake_train_loader,
            fake_validation_loader,
            fake_test_loader
        ),
    )

    fake_model = object()

    monkeypatch.setattr(
        train,
        "create_model",
        lambda config: fake_model,
    )

    trainer_calls = {}

    class FakeTrainer:

        def __init__(
            self,
            model,
            train_loader,
            eval_loader,
            config,
        ):

            trainer_calls["model"] = model
            trainer_calls["train_loader"] = (
                train_loader
            )
            trainer_calls["eval_loader"] = (
                eval_loader
            )

        def train(self):

            return {
                "history": [
                    {
                        "epoch": 1,
                        "train_loss": 0.5,
                        "accuracy": 0.8,
                        "macro_precision": 0.79,
                        "macro_recall": 0.78,
                        "macro_f1": 0.785,
                    }
                ],
                "best_metric": 0.785,
            }

        def evaluate(self, data_loader=None):

            assert data_loader is fake_test_loader

            return {
                "accuracy": 0.85,
                "macro_precision": 0.84,
                "macro_recall": 0.83,
                "macro_f1": 0.835
            }

    monkeypatch.setattr(
        train,
        "Trainer",
        FakeTrainer,
    )

    class FakePlotter:

        def __init__(
            self,
            history,
            output_dir,
        ):

            assert history
            assert output_dir == str(
                tmp_path / "reports"
            )

        def generate_all(self, metrics):

            assert metrics == [
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
            ]

            return {
                "training_loss": (
                    tmp_path
                    / "reports"
                    / "training_loss.png"
                ),
                "macro_f1": (
                    tmp_path
                    / "reports"
                    / "macro_f1.png"
                ),
            }

    monkeypatch.setattr(
        train,
        "TrainingPlotter",
        FakePlotter,
    )

    results = train.run_training(
        str(config_path)
    )

    assert results["best_metric"] == 0.785

    assert len(results["history"]) == 1

    assert "plots" in results

    assert (
        results["plots"]["training_loss"]
        .endswith("training_loss.png")
    )

    assert (
        trainer_calls["model"]
        is fake_model
    )

    assert (
        trainer_calls["train_loader"]
        is fake_train_loader
    )

    assert (
        trainer_calls["eval_loader"]
        is fake_validation_loader
    )

    assert results["test_metrics"]["macro_f1"] ==0.835



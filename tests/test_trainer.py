# using tiny synthetic dataset and a tiny mock model so we dont download distilbert or train banking77

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import Trainer


def build_config(tmp_path):
    return {
        "training": {
            "model_name": "distilbert-base-uncased",
            "num_labels": 3,
            "batch_size": 2,
            "eval_batch": 2,
            "epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "seed": 42,
            "device": "cpu",
            "output_dir": str(tmp_path / "models"),
            "checkpoint_dir": str(
                tmp_path / "checkpoints"
            ),
        },
        "evaluation": {
            "primary_metric": "macro_f1",
            "metrics": [
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
            ],
        },
    }


class TinyClassifier(torch.nn.Module):
    """Small classifier used only for trainer tests."""

    def __init__(
        self,
        vocab_size=20,
        hidden_size=8,
        num_labels=3,
    ):
        super().__init__()

        self.embedding = torch.nn.Embedding(
            vocab_size,
            hidden_size,
        )

        self.classifier = torch.nn.Linear(
            hidden_size,
            num_labels,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
    ):
        embeddings = self.embedding(input_ids)

        pooled = embeddings.mean(dim=1)

        logits = self.classifier(pooled)

        loss = None

        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits,
                labels,
            )

        return type(
            "ModelOutput",
            (),
            {
                "loss": loss,
                "logits": logits,
            },
        )()


def build_loaders():
    input_ids = torch.tensor(
        [
            [1, 2, 3],
            [2, 3, 4],
            [4, 5, 6],
            [5, 6, 7],
            [7, 8, 9],
            [8, 9, 10],
        ],
        dtype=torch.long,
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    labels = torch.tensor(
        [0, 1, 2, 0, 1, 2],
        dtype=torch.long,
    )

    dataset = TensorDataset(
        input_ids,
        attention_mask,
        labels,
    )

    def collate_fn(batch):
        input_ids, masks, labels = zip(*batch)

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.stack(labels),
        }

    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    eval_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, eval_loader


def build_trainer(tmp_path):
    config = build_config(tmp_path)

    train_loader, eval_loader = build_loaders()

    model = TinyClassifier(
        num_labels=3
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        config=config,
    )

    return trainer


def test_trainer_initialization(tmp_path):
    trainer = build_trainer(tmp_path)

    assert trainer.device.type == "cpu"
    assert trainer.epochs == 2
    assert trainer.learning_rate == 0.001
    assert trainer.primary_metric == "macro_f1"


def test_trainer_creates_directories(tmp_path):
    trainer = build_trainer(tmp_path)

    assert trainer.output_dir.exists()
    assert trainer.checkpoint_dir.exists()


def test_optimizer_is_initialized(tmp_path):
    trainer = build_trainer(tmp_path)

    assert trainer.optimizer is not None

    assert trainer.optimizer.param_groups[0][
        "lr"
    ] == 0.001


def test_scheduler_is_initialized(tmp_path):
    trainer = build_trainer(tmp_path)

    assert trainer.scheduler is not None


def test_train_epoch_returns_loss(tmp_path):
    trainer = build_trainer(tmp_path)

    loss = trainer.train_epoch()

    assert isinstance(loss, float)
    assert loss >= 0.0


def test_evaluate_returns_configured_metrics(tmp_path):
    trainer = build_trainer(tmp_path)

    metrics = trainer.evaluate()

    assert set(metrics.keys()) == {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    }

    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_save_checkpoint(tmp_path):
    trainer = build_trainer(tmp_path)

    metrics = {
        "accuracy": 0.5,
        "macro_precision": 0.5,
        "macro_recall": 0.5,
        "macro_f1": 0.5,
    }

    trainer.best_metric = 0.5

    checkpoint_path = trainer.save_checkpoint(
        epoch=1,
        metrics=metrics,
    )

    assert checkpoint_path.exists()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    assert checkpoint["epoch"] == 1
    assert checkpoint["best_metric"] == 0.5
    assert checkpoint["metrics"] == metrics
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "scheduler_state_dict" in checkpoint


def test_train_returns_history(tmp_path):
    trainer = build_trainer(tmp_path)

    results = trainer.train()

    assert "history" in results
    assert "best_metric" in results

    assert len(results["history"]) == 2

    for epoch_result in results["history"]:
        assert "epoch" in epoch_result
        assert "train_loss" in epoch_result
        assert "accuracy" in epoch_result
        assert "macro_precision" in epoch_result
        assert "macro_recall" in epoch_result
        assert "macro_f1" in epoch_result


def test_best_checkpoint_is_created(tmp_path):
    trainer = build_trainer(tmp_path)

    trainer.train()

    checkpoint_path = (
        trainer.checkpoint_dir
        / "best_model.pt"
    )

    assert checkpoint_path.exists()


def test_best_metric_is_macro_f1(tmp_path):
    trainer = build_trainer(tmp_path)

    trainer.train()

    assert trainer.best_metric >= 0.0
    assert trainer.best_metric <= 1.0
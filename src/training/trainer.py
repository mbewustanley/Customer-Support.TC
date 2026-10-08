"""
TrainingConfig
      │
      ▼
   Trainer
      │
      ├── model
      ├── train DataLoader
      ├── test DataLoader
      ├── optimizer
      ├── scheduler
      │
      ├── train_epoch()
      │
      ├── evaluate()
      │       └── ClassificationMetrics
      │
      ├── checkpoint_best_model()
      │       └── based on macro_f1
      │
      └── train()
              │
              ▼
       training artifacts"""


"""
Training loop:

for epoch:

    model.train()
    train batches
        ↓
    forward pass
        ↓
    loss
        ↓
    backward
        ↓
    optimizer step
        ↓
    scheduler step

    
    model.eval()
        ↓
    evaluate on test set
        ↓
    calculate metrics
        ↓
    compare macro_f1
        ↓
    save best checkpoint"""



"""
                    ┌──────────────────┐
                    │ training_config  │
                    └────────┬─────────┘
                             │
                             ▼
┌─────────────┐      ┌──────────────┐      ┌────────────────────┐
│ Banking77   │ ───► │ DataLoader   │ ───► │     Trainer        │
│ Dataset     │      └──────────────┘      │                    │
└─────────────┘                            │ train_epoch()      │
                                           │ evaluate()         │
┌─────────────┐                            │ checkpoint()       │
│ Classifier  │ ─────────────────────────► │ train()            │
└─────────────┘                            └─────────┬──────────┘
                                                    │
                                                    ▼
                                           ClassificationMetrics
                                                    │
                                                    ▼
                                      accuracy / precision /
                                      recall / macro_f1
"""




from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.evaluation.metrics import ClassificationMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Handles model training, evaluation, checkpointing,
    and training history for the Banking77 classifier.
    """

## every mention of eval in script refers to validation
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        config: Dict[str, Any],
    ) -> None:

        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config

        training_config = config["training"]
        evaluation_config = config["evaluation"]

        self.epochs = training_config["epochs"]
        self.learning_rate = training_config["learning_rate"]
        self.weight_decay = training_config["weight_decay"]
        self.warmup_ratio = training_config["warmup_ratio"]
        self.seed = training_config["seed"]

        self.output_dir = Path(
            training_config["output_dir"]
        )

        self.checkpoint_dir = Path(
            training_config["checkpoint_dir"]
        )

        self.primary_metric = evaluation_config[
            "primary_metric"
        ]

        self.metric_names = evaluation_config["metrics"]

        self._set_seed(self.seed)

        self.device = self._resolve_device(
            training_config["device"]
        )

        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps = (
            len(self.train_loader) * self.epochs
        )

        warmup_steps = int(
            total_steps * self.warmup_ratio
        )

        self.scheduler = self._create_scheduler(
            optimizer=self.optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )

        self.best_metric = float("-inf")

        self.history: List[Dict[str, Any]] = []

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Trainer initialized."
        )

        logger.info(
            "Training device: %s",
            self.device,
        )

        logger.info(
            "Training epochs: %d",
            self.epochs,
        )

        logger.info(
            "Total training steps: %d",
            total_steps,
        )

        logger.info(
            "Warmup steps: %d",
            warmup_steps,
        )

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Set random seeds for reproducibility."""

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _resolve_device(
        device: str,
    ) -> torch.device:
        """Resolve configured device."""

        if device == "auto":
            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        return torch.device(device)

    @staticmethod
    def _create_scheduler(
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
    ) -> LambdaLR:
        """Create linear learning-rate scheduler with warmup."""

        def lr_lambda(
            current_step: int,
        ) -> float:

            if current_step < warmup_steps:

                if warmup_steps == 0:
                    return 1.0

                return (
                    float(current_step)
                    / float(max(1, warmup_steps))
                )

            remaining_steps = (
                total_steps - current_step
            )

            decay_steps = (
                total_steps - warmup_steps
            )

            if decay_steps <= 0:
                return 1.0

            return max(
                0.0,
                float(remaining_steps)
                / float(decay_steps),
            )

        return LambdaLR(
            optimizer,
            lr_lambda,
        )

    def train_epoch(self) -> float:
        """
        Train the model for one epoch.

        Returns:
            Average training loss.
        """

        self.model.train()

        total_loss = 0.0

        for batch in self.train_loader:

            input_ids = batch["input_ids"].to(
                self.device
            )

            attention_mask = batch[
                "attention_mask"
            ].to(self.device)

            labels = batch["labels"].to(
                self.device
            )

            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            loss.backward()

            self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(self.train_loader)
        )

        logger.info(
            "Training loss: %.4f",
            average_loss,
        )

        return average_loss

    def evaluate(self, data_loader=None) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.

        Returns:
            Dictionary containing configured metrics.
        """

        if data_loader is None:
            data_loader = self.eval_loader

        self.model.eval()

        predictions: List[int] = []
        labels: List[int] = []

        with torch.no_grad():

            for batch in self.eval_loader:

                input_ids = batch[
                    "input_ids"
                ].to(self.device)

                attention_mask = batch[
                    "attention_mask"
                ].to(self.device)

                batch_labels = batch[
                    "labels"
                ].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                batch_predictions = torch.argmax(
                    outputs.logits,
                    dim=-1,
                )

                predictions.extend(
                    batch_predictions.cpu().tolist()
                )

                labels.extend(
                    batch_labels.cpu().tolist()
                )

        metrics = ClassificationMetrics.compute(
            predictions=predictions,
            labels=labels,
        )

        selected_metrics = {
            name: metrics[name]
            for name in self.metric_names
        }

        logger.info(
            "Evaluation metrics: %s",
            selected_metrics,
        )

        return selected_metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
    ) -> Path:
        """
        Save the current best model checkpoint.

        Returns:
            Path to saved checkpoint.
        """

        checkpoint_path = (
            self.checkpoint_dir
            / "best_model.pt"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": (
                self.model.state_dict()
            ),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
            ),
            "metrics": metrics,
            "best_metric": self.best_metric,
            "config": self.config,
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        logger.info(
            "Best checkpoint saved to %s",
            checkpoint_path,
        )

        return checkpoint_path

    def train(self) -> Dict[str, Any]:
        """
        Run the complete training process.

        Returns:
            Training history and best evaluation results.
        """

        logger.info(
            "Starting model training."
        )

        for epoch in range(1, self.epochs + 1):

            logger.info(
                "Starting epoch %d/%d",
                epoch,
                self.epochs,
            )

            train_loss = self.train_epoch()

            metrics = self.evaluate()

            epoch_results = {
                "epoch": epoch,
                "train_loss": train_loss,
                **metrics,
            }

            self.history.append(
                epoch_results
            )

            current_metric = metrics[
                self.primary_metric
            ]

            if current_metric > self.best_metric:

                self.best_metric = current_metric

                self.save_checkpoint(
                    epoch=epoch,
                    metrics=metrics,
                )

                logger.info(
                    "New best %s: %.4f",
                    self.primary_metric,
                    current_metric,
                )

        logger.info(
            "Training completed. Best %s: %.4f",
            self.primary_metric,
            self.best_metric,
        )

        return {
            "history": self.history,
            "best_metric": self.best_metric,
        }
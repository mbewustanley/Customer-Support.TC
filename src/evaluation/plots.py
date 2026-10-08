from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingPlotter:
    """
    Generate training and evaluation plots from trainer history.
    """

    def __init__(
        self,
        history: List[Dict],
        output_dir: str = "artifacts/reports",
    ) -> None:

        if not isinstance(history, list):
            raise TypeError("Training history must be a list.")

        if not history:
            raise ValueError("Training history cannot be empty.")

        self.history = history
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Initialized TrainingPlotter with %d epochs.",
            len(history),
        )

    def _get_epochs(self) -> List[int]:
        """Extract epoch numbers from training history."""

        return [
            record["epoch"]
            for record in self.history
        ]

    def _get_metric_values(
        self,
        metric_name: str,
    ) -> List[float]:
        """Extract a metric from every epoch."""

        values = []

        for record in self.history:

            if metric_name not in record:
                raise KeyError(
                    f"Metric '{metric_name}' "
                    "not found in training history."
                )

            values.append(
                float(record[metric_name])
            )

        return values

    def plot_training_loss(self) -> Path:
        """Plot training loss against epoch."""

        epochs = self._get_epochs()
        losses = self._get_metric_values("train_loss")

        output_path = (
            self.output_dir / "training_loss.png"
        )

        plt.figure()

        plt.plot(
            epochs,
            losses,
            marker="o",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Training Loss")

        plt.xticks(epochs)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        logger.info(
            "Training loss plot saved to %s.",
            output_path,
        )

        return output_path

    def plot_metric(
        self,
        metric_name: str,
    ) -> Path:
        """Plot a single evaluation metric against epoch."""

        epochs = self._get_epochs()
        values = self._get_metric_values(
            metric_name
        )

        output_path = (
            self.output_dir / f"{metric_name}.png"
        )

        plt.figure()

        plt.plot(
            epochs,
            values,
            marker="o",
        )

        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(
            f"{metric_name} by Epoch"
        )

        plt.xticks(epochs)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        logger.info(
            "%s plot saved to %s.",
            metric_name,
            output_path,
        )

        return output_path

    def plot_all_metrics(
        self,
        metrics: List[str],
    ) -> Path:
        """Plot all evaluation metrics on one figure."""

        if not metrics:
            raise ValueError(
                "Metrics list cannot be empty."
            )

        epochs = self._get_epochs()

        output_path = (
            self.output_dir
            / "evaluation_metrics.png"
        )

        plt.figure()

        for metric_name in metrics:

            values = self._get_metric_values(
                metric_name
            )

            plt.plot(
                epochs,
                values,
                marker="o",
                label=metric_name,
            )

        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(
            "Evaluation Metrics by Epoch"
        )

        plt.xticks(epochs)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        logger.info(
            "Evaluation metrics plot saved to %s.",
            output_path,
        )

        return output_path

    def generate_all(
        self,
        metrics: List[str],
    ) -> Dict[str, Path]:
        """
        Generate the complete training report set.
        """

        plots = {}

        plots["training_loss"] = (
            self.plot_training_loss()
        )

        for metric_name in metrics:
            plots[metric_name] = (
                self.plot_metric(metric_name)
            )

        plots["evaluation_metrics"] = (
            self.plot_all_metrics(metrics)
        )

        logger.info(
            "All training plots generated."
        )

        return plots
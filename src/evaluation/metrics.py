# Metrics API
"""
ClassificationMetrics
accuracy, macro_precision, macro_recall, macro_f1, 
compute()

input: predictions, labels
output:
{
"accuracy": ...
...
}"""


from typing import Dict, Sequence

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ClassificationMetrics:
    """Compute classification metrics for model evaluation."""

    @staticmethod
    def compute(
        predictions: Sequence[int],
        labels: Sequence[int],
    ) -> Dict[str, float]:
        """
        Compute accuracy and macro-averaged classification metrics.

        Args:
            predictions: Predicted class labels.
            labels: Ground-truth class labels.

        Returns:
            Dictionary containing accuracy, macro precision,
            macro recall, and macro F1.
        """

        if len(predictions) != len(labels):
            raise ValueError(
                "Predictions and labels must have the same length."
            )

        if len(predictions) == 0:
            raise ValueError(
                "Predictions and labels cannot be empty."
            )

        metrics = {
            "accuracy": float(
                accuracy_score(labels, predictions)
            ),
            "macro_precision": float(
                precision_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_recall": float(
                recall_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_f1": float(
                f1_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
        }

        logger.info(
            "Classification metrics computed: %s",
            metrics,
        )

        return metrics
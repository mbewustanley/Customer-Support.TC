from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Banking77Classifier(torch.nn.Module):

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 77,
    ) -> None:
        super().__init__()

        if num_labels <= 0:
            raise ValueError(
                "num_labels must be greater than zero."
            )

        self.model_name = model_name
        self.num_labels = num_labels

        logger.info(
            "Loading model '%s' with %d labels.",
            model_name,
            num_labels,
        )

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
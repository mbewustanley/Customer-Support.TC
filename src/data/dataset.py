from typing import Dict

import torch
from datasets import Dataset
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from src.utils.logger import get_logger


logger = get_logger(__name__)


class Banking77Dataset(TorchDataset):
    """
    PyTorch Dataset wrapper around the tokenized
    Banking77 Hugging Face dataset.
    """

    REQUIRED_COLUMNS = {
        "input_ids",
        "attention_mask",
        "label",
    }

    def __init__(
        self,
        dataset: Dataset,
    ) -> None:

        self._validate_dataset(dataset)

        self.dataset = dataset

        logger.info(
            "Initialized Banking77Dataset with %d samples.",
            len(self.dataset),
        )

    @classmethod
    def _validate_dataset(
        cls,
        dataset: Dataset,
    ) -> None:
        """Validate the tokenized dataset schema."""

        if not isinstance(dataset, Dataset):
            raise TypeError(
                "dataset must be a Hugging Face "
                "datasets.Dataset."
            )

        columns = set(dataset.column_names)

        missing_columns = (
            cls.REQUIRED_COLUMNS - columns
        )

        if missing_columns:
            raise ValueError(
                "Tokenized dataset is missing "
                "required columns: "
                f"{sorted(missing_columns)}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__( self, index: int ) -> Dict[str, Tensor]:

        item = self.dataset[index]

        return {
            "input_ids": torch.tensor(
                item["input_ids"],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                item["attention_mask"],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                item["label"],
                dtype=torch.long,
            ),
        }
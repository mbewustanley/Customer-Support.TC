"""
The tokenizer will:

load data/processed/train and test
load the configured Hugging Face tokenizer
tokenize only the text column
preserve label
use truncation + padding
default to max_length=128
save Hugging Face datasets to data/tokenized/train and data/tokenized/test
return the tokenized datasets
fail clearly if the expected schema is missing
"""



from pathlib import Path
from typing import Tuple

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Banking77Tokenizer:
    """
    Tokenizes the processed Banking77 dataset using
    a Hugging Face tokenizer.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        processed_data_dir: str = "data/processed",
        tokenized_data_dir: str = "data/tokenized",
        max_length: int = 128,
    ) -> None:

        if max_length <= 0:
            raise ValueError(
                "max_length must be greater than zero."
            )

        self.model_name = model_name
        self.processed_data_dir = Path(
            processed_data_dir
        )
        self.tokenized_data_dir = Path(
            tokenized_data_dir
        )
        self.max_length = max_length

        logger.info(
            "Loading tokenizer: %s",
            self.model_name,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

    def load_data(
        self,
    ) -> Tuple[Dataset, Dataset]:
        """Load processed train and test datasets."""

        train_path = (
            self.processed_data_dir / "train"
        )
        test_path = (
            self.processed_data_dir / "test"
        )

        if not train_path.exists():
            raise FileNotFoundError(
                f"Processed train dataset not found: "
                f"{train_path}"
            )

        if not test_path.exists():
            raise FileNotFoundError(
                f"Processed test dataset not found: "
                f"{test_path}"
            )

        train_dataset = load_from_disk(
            str(train_path)
        )
        test_dataset = load_from_disk(
            str(test_path)
        )

        return train_dataset, test_dataset

    @staticmethod
    def _validate_dataset_schema(
        dataset: Dataset,
    ) -> None:
        """Validate the columns required for tokenization."""

        required_columns = {
            "text",
            "label",
        }

        missing_columns = (
            required_columns
            - set(dataset.column_names)
        )

        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

    def tokenize_dataset(
        self,
        dataset: Dataset,
    ) -> Dataset:
        """
        Tokenize a dataset while preserving labels.
        """

        self._validate_dataset_schema(dataset)

        logger.info(
            "Tokenizing %d records with max_length=%d.",
            len(dataset),
            self.max_length,
        )

        def tokenize_batch(batch):
            return self.tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        tokenized_dataset = dataset.map(
            tokenize_batch,
            batched=True,
            desc="Tokenizing Banking77 dataset",
        )

        return tokenized_dataset

    def save_dataset(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
    ) -> None:
        """Save tokenized train and test datasets."""

        self.tokenized_data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_path = (
            self.tokenized_data_dir / "train"
        )
        test_path = (
            self.tokenized_data_dir / "test"
        )

        train_dataset.save_to_disk(
            str(train_path)
        )

        test_dataset.save_to_disk(
            str(test_path)
        )

        logger.info(
            "Tokenized datasets saved to %s",
            self.tokenized_data_dir,
        )

    def run(
        self,
    ) -> Tuple[Dataset, Dataset]:
        """Run the complete tokenization pipeline."""

        train_dataset, test_dataset = (
            self.load_data()
        )

        train_tokenized = self.tokenize_dataset(
            train_dataset
        )

        test_tokenized = self.tokenize_dataset(
            test_dataset
        )

        self.save_dataset(
            train_tokenized,
            test_tokenized,
        )

        return (
            train_tokenized,
            test_tokenized,
        )


if __name__ == "__main__":
    tokenizer = Banking77Tokenizer()
    tokenizer.run()
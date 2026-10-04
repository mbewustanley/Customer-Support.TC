# this file is used to preprocess the data before tokenization with huggingface.
#  It will clean the data and convert it into a format that can be used by the model.

from pathlib import Path
from typing import Tuple

from datasets import Dataset, load_from_disk

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.exceptions import CustomException


logger = get_logger(__name__)

# create preprocessor class
class Banking77Preprocessor:
    """preprocess the validated Banking77 dataset """

    def __init__(
            self,
            config_path: str = "configs/data_config.yaml",
            raw_data_dir: str = "data/raw/",
            processed_data_dir: str = "data/processed/",
            ):

        try:
            self.config = load_config(config_path)
            logger.info(f"Loaded config from {config_path}")

            self.raw_data_dir = Path(raw_data_dir)
            self.processed_data_dir = Path(processed_data_dir)

            # create processed data directory if it doesn't exist
            self.processed_data_dir.mkdir(parents=True, exist_ok=True)


            self.text_column = "text"
            self.label_column = "label"

            self.num_classes = self.config["preprocessing"]["expected_num_classes"]

            logger.info(f"Initialized Banking77Preprocessor with raw_data_dir: {self.raw_data_dir}, processed_data_dir: {self.processed_data_dir}, num_classes: {self.num_classes}")

        except Exception as e:
            logger.error(f"Error initializing Banking77Preprocessor: {e}")
            raise CustomException(f"Error initializing Banking77Preprocessor: {e}")



    def load_data(self)  -> Tuple[Dataset, Dataset]:
        """Load the raw data from disk and return train and test datasets."""

        try:
            logger.info(f"Loading train and test datasets from {self.raw_data_dir}")

            train_data = load_from_disk(self.raw_data_dir / "train")
            test_data = load_from_disk(self.raw_data_dir / "test")

            logger.info(f"Loaded train({len(train_data)}) and test({len(test_data)}) datasets from {self.raw_data_dir}")

            return train_data, test_data

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException(f"Error loading data: {e}")



    def clean_text(self, dataset: Dataset) -> Dataset:
        """Clean the text data by stripping whitespace."""

        try:
            logger.info(f"Cleaning text data in dataset with {len(dataset)} samples")

            if not self.config["preprocessing"]["strip_whitespace"]:
                return dataset

            def strip_text(example):
                example[self.text_column] = example[self.text_column].strip()
                return example

            return dataset.map(strip_text, desc="Stripping whitespace from text data")

        except Exception as e:
            logger.error(f"Error cleaning text data: {e}")
            raise CustomException(f"Error cleaning text data: {e}")
        

    
    def remove_invalid_records(self, dataset: Dataset) -> Dataset:
        """Remove records with empty text or invalid labels."""

        try:
            logger.info(f"Removing invalid records from dataset with {len(dataset)} samples")

            if not self.config["preprocessing"]["remove_empty_text"]:
                return dataset


            original_length = len(dataset)

            logger.info("checking for empty text records")

            dataset = dataset.filter(
                lambda x: (
                    x[self.text_column] is not None and 
                    isinstance(x[self.text_column], str) and 
                    len(x[self.text_column].strip()) > 0
                ),
                desc="Removing records with empty text"
            )
            removed = original_length - len(dataset)
            logger.info(f"Removed {removed} records with empty text")

            return dataset
        
        except Exception as e:  
            logger.error(f"Error removing invalid records: {e}")
            raise CustomException(f"Error removing invalid records: {e}")


    def normalize_labels(self, dataset: Dataset) -> Dataset:
        """Normalize the labels to be in the range [0, num_classes-1]."""

        try:
            logger.info(f"Normalizing labels in dataset with {len(dataset)} samples")

            if not self.config["preprocessing"]["normalize_labels"]:
                return dataset

            def normalize_label(example):
                example[self.label_column] = int(example[self.label_column])
                return example

            dataset = dataset.map(normalize_label, desc="Normalizing labels")

            # check if all labels are in the expected range
            unique_labels = set(dataset[self.label_column])
            if not all(0 <= label < self.num_classes for label in unique_labels):
                raise CustomException(f"Labels are not in the expected range [0, {self.num_classes-1}]. Found labels: {unique_labels}")

            logger.info(f"Normalized labels to be in the range [0, {self.num_classes-1}]")

            return dataset

        except Exception as e:
            logger.error(f"Error normalizing labels: {e}")
            raise CustomException(f"Error normalizing labels: {e}")




    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Preprocess the dataset by cleaning text, removing invalid records, and normalizing labels."""

        try:
            logger.info(f"Starting preprocessing of dataset with {len(dataset)} samples")

            dataset = self.clean_text(dataset)
            dataset = self.remove_invalid_records(dataset)
            dataset = self.normalize_labels(dataset)

            logger.info(f"Completed preprocessing of dataset. Final size: {len(dataset)} samples")

            return dataset

        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise CustomException(f"Error during preprocessing: {e}")


    def save_dataset(self, 
                     train_dataset: Dataset, 
                     test_dataset: Dataset,) -> None:
        """Save the preprocessed dataset to disk."""

        try:
            train_path = self.processed_data_dir / "train"
            train_dataset.save_to_disk(train_path)
            logger.info(f"Saved preprocessed train dataset with {len(train_dataset)} samples to {train_path}")

            test_path = self.processed_data_dir / "test"
            test_dataset.save_to_disk(test_path)
            logger.info(f"Saved preprocessed test dataset with {len(test_dataset)} samples to {test_path}")

            
        except Exception as e:
            logger.error(f"Error saving datasets: {e}")
            raise CustomException(f"Error saving datasets: {e}")


    def run(self) -> Tuple[Dataset, Dataset]:
        """Run the preprocessing pipeline: load data, preprocess, and save."""

        try:
            logger.info("Starting preprocessing pipeline")

            train_dataset, test_dataset = self.load_data()

            train_dataset = self.preprocess_dataset(train_dataset)
            test_dataset = self.preprocess_dataset(test_dataset)

            self.save_dataset(train_dataset, test_dataset)

            logger.info("Completed preprocessing pipeline")

            return train_dataset, test_dataset

        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            raise CustomException(f"Error in preprocessing pipeline: {e}")


if __name__ == "__main__":
    preprocessor = Banking77Preprocessor()
    preprocessor.run()
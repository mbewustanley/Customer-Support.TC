from pathlib import Path

from datasets import load_from_disk

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.exceptions import CustomException
import sys


logger = get_logger(__name__)

"""
                    configs/
                       │
                data_config.yaml
                       │
                       ▼
                config_loader.py
                       │
                       ▼
data/raw ──────► validation.py
                       │
              ┌────────┴────────┐
              │                 │
          logger.py        exceptions.py
              │                 │
              └────────┬────────┘
                       ▼
                  VALID / FAIL
                       │
                       ▼
                 preprocessing"""


class Banking77Validator:
    """
    Validator for the ingested PolyAI Banking77 dataset.

    The validator checks:
        - Dataset directories
        - Required dataset files
        - Dataset loading
        - Expected columns
        - Data types
        - Train/test schema consistency
        - Expected row counts
        - Missing values
        - Empty strings
        - Whitespace-only strings
        - Duplicate rows
        - Duplicate text values
        - Label data types
        - Label ranges
        - Expected number of classes
        - Missing classes
        - Class distribution
        - Train/test text leakage
        - Train/test split ratio
    """

    def __init__(
        self,
        config_path: str = "configs/data_config.yaml",
        raw_data_dir: str = "data/raw",
    ):
        """
        Initialize the Banking77 validator.

        Args:
            config_path: Path to the YAML data configuration.
            raw_data_dir: Directory containing the ingested dataset.
        """

        self.config = load_config(config_path)

        self.raw_data_dir = Path(raw_data_dir)
        self.train_dir = self.raw_data_dir / "train"
        self.test_dir = self.raw_data_dir / "test"

        self.train_dataset = None
        self.test_dataset = None

        self.errors = []
        self.warnings = []

        # --------------------------------------------------------------
        # Configuration
        # --------------------------------------------------------------

        self.expected_columns = self.config["schema"]["columns"]

        self.expected_train_rows = self.config["size"]["train"]
        self.expected_test_rows = self.config["size"]["test"]

        self.num_classes = self.config["classification"]["num_classes"]
        self.label_indexing = self.config["classification"]["label_indexing"]

        self.split_tolerance = self.config["split"]["tolerance"]

        self.min_text_length = self.config["text"]["min_length"]

    # ==================================================================
    # Helper methods
    # ==================================================================

    def _add_error(self, message: str) -> None:
        """Record a validation error."""

        self.errors.append(message)
        logger.error(message)

    def _add_warning(self, message: str) -> None:
        """Record a validation warning."""

        self.warnings.append(message)
        logger.warning(message)

    # ==================================================================
    # Directory validation
    # ==================================================================

    def validate_directories(self) -> None:
        """Validate that the expected raw data directories exist."""

        logger.info("Validating raw dataset directories.")

        if not self.raw_data_dir.exists():
            self._add_error(
                f"Raw data directory does not exist: "
                f"{self.raw_data_dir}"
            )
            return

        if not self.train_dir.exists():
            self._add_error(
                f"Training dataset directory does not exist: "
                f"{self.train_dir}"
            )

        if not self.test_dir.exists():
            self._add_error(
                f"Test dataset directory does not exist: "
                f"{self.test_dir}"
            )

    # ==================================================================
    # Dataset file validation
    # ==================================================================

    def validate_dataset_files(self) -> None:
        """
        Validate the expected Hugging Face dataset files.

        Each split should contain:
            - At least one .arrow file
            - dataset_info.json
            - state.json
        """

        logger.info("Validating dataset files.")

        splits = {
            "train": self.train_dir,
            "test": self.test_dir,
        }

        for split_name, split_dir in splits.items():

            if not split_dir.exists():
                continue

            arrow_files = list(split_dir.glob("*.arrow"))

            if not arrow_files:
                self._add_error(
                    f"{split_name}: No .arrow file found in "
                    f"{split_dir}"
                )

            dataset_info = split_dir / "dataset_info.json"

            if not dataset_info.exists():
                self._add_error(
                    f"{split_name}: dataset_info.json not found."
                )

            state_file = split_dir / "state.json"

            if not state_file.exists():
                self._add_error(
                    f"{split_name}: state.json not found."
                )

    # ==================================================================
    # Dataset loading
    # ==================================================================

    def load_datasets(self) -> None:
        """Load the train and test datasets from disk."""

        logger.info("Loading Banking77 datasets.")

        try:

            self.train_dataset = load_from_disk(
                str(self.train_dir)
            )

            self.test_dataset = load_from_disk(
                str(self.test_dir)
            )

            logger.info(
                "Datasets loaded successfully."
            )

            logger.info(
                "Train samples: %d",
                len(self.train_dataset),
            )

            logger.info(
                "Test samples: %d",
                len(self.test_dataset),
            )

        except Exception as error:

            raise CustomException(
                error,
                sys,
            )

    # ==================================================================
    # Column validation
    # ==================================================================

    def validate_columns(
        self,
        dataset,
        split_name: str,
    ) -> None:
        """Validate expected and unexpected columns."""

        logger.info(
            "Validating columns for %s dataset.",
            split_name,
        )

        expected_columns = set(
            self.expected_columns.keys()
        )

        actual_columns = set(
            dataset.column_names
        )

        missing_columns = (
            expected_columns - actual_columns
        )

        unexpected_columns = (
            actual_columns - expected_columns
        )

        if missing_columns:

            self._add_error(
                f"{split_name}: Missing columns: "
                f"{sorted(missing_columns)}"
            )

        if unexpected_columns:

            self._add_error(
                f"{split_name}: Unexpected columns: "
                f"{sorted(unexpected_columns)}"
            )

    # ==================================================================
    # Data type validation
    # ==================================================================

    def validate_dtypes(
        self,
        dataset,
        split_name: str,
    ) -> None:
        """Validate expected data types."""

        logger.info(
            "Validating data types for %s dataset.",
            split_name,
        )

        for column, expected_type in self.expected_columns.items():

            if column not in dataset.column_names:
                continue

            actual_type = str(
                dataset.features[column]
            )

            expected_feature_type = (
                f"Value('{expected_type}')"
            )

            if actual_type != expected_feature_type:

                self._add_error(
                    f"{split_name}: Column '{column}' "
                    f"has type {actual_type}; "
                    f"expected {expected_feature_type}."
                )

    # ==================================================================
    # Schema consistency
    # ==================================================================

    def validate_schema_consistency(self) -> None:
        """Ensure train and test datasets have the same schema."""

        logger.info(
            "Validating train/test schema consistency."
        )

        if (
            self.train_dataset is None
            or self.test_dataset is None
        ):
            return

        if (
            self.train_dataset.column_names
            != self.test_dataset.column_names
        ):

            self._add_error(
                "Train and test datasets have different "
                "column schemas."
            )

        if (
            self.train_dataset.features
            != self.test_dataset.features
        ):

            self._add_error(
                "Train and test datasets have different "
                "feature schemas."
            )

    # ==================================================================
    # Row count validation
    # ==================================================================

    def validate_row_counts(self) -> None:
        """Validate expected train and test dataset sizes."""

        logger.info("Validating dataset row counts.")

        if (
            self.train_dataset is None
            or self.test_dataset is None
        ):
            return

        actual_train_rows = len(
            self.train_dataset
        )

        actual_test_rows = len(
            self.test_dataset
        )

        if actual_train_rows != self.expected_train_rows:

            self._add_error(
                f"Train dataset contains "
                f"{actual_train_rows:,} rows; "
                f"expected {self.expected_train_rows:,}."
            )

        if actual_test_rows != self.expected_test_rows:

            self._add_error(
                f"Test dataset contains "
                f"{actual_test_rows:,} rows; "
                f"expected {self.expected_test_rows:,}."
            )

    # ==================================================================
    # Missing value validation
    # ==================================================================

    def validate_missing_values(
        self,
        dataset,
        split_name: str,
    ) -> None:
        """Check expected columns for missing values."""

        logger.info(
            "Checking missing values in %s dataset.",
            split_name,
        )

        for column in self.expected_columns:

            if column not in dataset.column_names:
                continue

            missing_count = sum(
                value is None
                for value in dataset[column]
            )

            if missing_count > 0:

                self._add_error(
                    f"{split_name}: Column '{column}' "
                    f"contains {missing_count:,} "
                    f"missing values."
                )

    # ==================================================================
    # Text validation
    # ==================================================================

    def validate_text(
        self,
        dataset,
        split_name: str,
    ) -> None:
        """Validate text values."""

        logger.info(
            "Validating text values in %s dataset.",
            split_name,
        )

        if "text" not in dataset.column_names:
            return

        texts = dataset["text"]

        empty_strings = 0
        whitespace_strings = 0
        invalid_types = 0
        short_texts = 0

        for text in texts:

            if text is None:
                continue

            if not isinstance(text, str):

                invalid_types += 1
                continue

            if text == "":
                empty_strings += 1

            if text.strip() == "":
                whitespace_strings += 1

            if len(text.strip()) < self.min_text_length:
                short_texts += 1

        if empty_strings > 0:

            self._add_error(
                f"{split_name}: Found "
                f"{empty_strings:,} empty strings."
            )

        if whitespace_strings > 0:

            self._add_error(
                f"{split_name}: Found "
                f"{whitespace_strings:,} "
                f"whitespace-only strings."
            )

        if invalid_types > 0:

            self._add_error(
                f"{split_name}: Found "
                f"{invalid_types:,} text values "
                f"with invalid types."
            )

        if short_texts > 0:

            self._add_error(
                f"{split_name}: Found "
                f"{short_texts:,} text values shorter "
                f"than {self.min_text_length} character(s)."
            )

    # ==================================================================
    # Duplicate validation
    # ==================================================================

    def validate_duplicates(
        self,
        dataset,
        split_name: str,
    ) -> None:
        """Check for duplicate rows and duplicate text."""

        logger.info(
            "Checking duplicates in %s dataset.",
            split_name,
        )

        if len(dataset) == 0:
            return

        texts = dataset["text"]
        labels = dataset["label"]

        # --------------------------------------------------------------
        # Complete row duplicates
        # --------------------------------------------------------------

        rows = list(
            zip(texts, labels)
        )

        duplicate_rows = (
            len(rows) - len(set(rows))
        )

        if duplicate_rows > 0:

            self._add_warning(
                f"{split_name}: Found "
                f"{duplicate_rows:,} duplicate rows."
            )

        # --------------------------------------------------------------
        # Duplicate text
        # --------------------------------------------------------------

        duplicate_texts = (
            len(texts) - len(set(texts))
        )

        if duplicate_texts > 0:

            self._add_warning(
                f"{split_name}: Found "
                f"{duplicate_texts:,} duplicate "
                f"text values."
            )

    # ==================================================================
    # Label validation
    # ==================================================================

    def validate_labels(
        self,
        dataset,
        split_name: str,
    ) -> None:
        """Validate label types, ranges and class count."""

        logger.info(
            "Validating labels in %s dataset.",
            split_name,
        )

        if "label" not in dataset.column_names:
            return

        labels = dataset["label"]

        if not labels:

            self._add_error(
                f"{split_name}: Label column is empty."
            )
            return

        # --------------------------------------------------------------
        # Label datatype
        # --------------------------------------------------------------

        invalid_types = [
            label
            for label in labels
            if not isinstance(label, int)
        ]

        if invalid_types:

            self._add_error(
                f"{split_name}: Found "
                f"{len(invalid_types):,} non-integer labels."
            )

            return

        # --------------------------------------------------------------
        # Determine expected range
        # --------------------------------------------------------------

        if self.label_indexing == "zero_based":

            min_label = 0
            max_label = self.num_classes - 1

        else:

            self._add_error(
                f"Unsupported label indexing strategy: "
                f"{self.label_indexing}"
            )

            return

        # --------------------------------------------------------------
        # Range validation
        # --------------------------------------------------------------

        actual_min = min(labels)
        actual_max = max(labels)

        if actual_min != min_label:

            self._add_error(
                f"{split_name}: Minimum label is "
                f"{actual_min}; expected {min_label}."
            )

        if actual_max != max_label:

            self._add_error(
                f"{split_name}: Maximum label is "
                f"{actual_max}; expected {max_label}."
            )

        invalid_labels = [
            label
            for label in labels
            if label < min_label
            or label > max_label
        ]

        if invalid_labels:

            self._add_error(
                f"{split_name}: Found "
                f"{len(invalid_labels):,} labels "
                f"outside the valid range "
                f"[{min_label}, {max_label}]."
            )

        # --------------------------------------------------------------
        # Number of classes
        # --------------------------------------------------------------

        unique_labels = set(labels)

        if len(unique_labels) != self.num_classes:

            self._add_error(
                f"{split_name}: Found "
                f"{len(unique_labels)} unique classes; "
                f"expected {self.num_classes}."
            )

        # --------------------------------------------------------------
        # Missing classes
        # --------------------------------------------------------------

        expected_labels = set(
            range(min_label, max_label + 1)
        )

        missing_labels = (
            expected_labels - unique_labels
        )

        if missing_labels:

            self._add_error(
                f"{split_name}: Missing label classes: "
                f"{sorted(missing_labels)}"
            )

    # ==================================================================
    # Train/test leakage
    # ==================================================================

    def validate_train_test_leakage(self) -> None:
        """
        Check whether identical text appears in both train and test.

        Identical samples across train and test can result in
        evaluation leakage.
        """

        logger.info(
            "Checking for train/test data leakage."
        )

        if (
            self.train_dataset is None
            or self.test_dataset is None
        ):
            return

        train_texts = set(
            self.train_dataset["text"]
        )

        test_texts = set(
            self.test_dataset["text"]
        )

        overlap = train_texts.intersection(
            test_texts
        )

        if overlap:

            self._add_error(
                f"Train/test leakage detected: "
                f"{len(overlap):,} identical text samples "
                f"appear in both datasets."
            )

    # ==================================================================
    # Split ratio validation
    # ==================================================================

    def validate_split_ratio(self) -> None:
        """Validate the actual train/test split ratio."""

        logger.info(
            "Validating train/test split ratio."
        )

        if (
            self.train_dataset is None
            or self.test_dataset is None
        ):
            return

        train_rows = len(
            self.train_dataset
        )

        test_rows = len(
            self.test_dataset
        )

        total_rows = train_rows + test_rows

        if total_rows == 0:

            self._add_error(
                "Train and test datasets contain zero rows."
            )

            return

        actual_train_ratio = (
            train_rows / total_rows
        )

        actual_test_ratio = (
            test_rows / total_rows
        )

        expected_train_ratio = (
            self.expected_train_rows
            / (
                self.expected_train_rows
                + self.expected_test_rows
            )
        )

        expected_test_ratio = (
            self.expected_test_rows
            / (
                self.expected_train_rows
                + self.expected_test_rows
            )
        )

        train_difference = abs(
            actual_train_ratio
            - expected_train_ratio
        )

        test_difference = abs(
            actual_test_ratio
            - expected_test_ratio
        )

        if train_difference > self.split_tolerance:

            self._add_error(
                f"Train split ratio is "
                f"{actual_train_ratio:.4f}; "
                f"expected approximately "
                f"{expected_train_ratio:.4f}."
            )

        if test_difference > self.split_tolerance:

            self._add_error(
                f"Test split ratio is "
                f"{actual_test_ratio:.4f}; "
                f"expected approximately "
                f"{expected_test_ratio:.4f}."
            )

    # ==================================================================
    # Class distribution
    # ==================================================================

    def validate_class_distribution(
        self,
        dataset,
        split_name: str,
    ) -> None:
        """
        Check that every expected class is represented.

        This is separate from validate_labels() so that class
        distribution validation can be expanded later.
        """

        logger.info(
            "Validating class distribution for %s dataset.",
            split_name,
        )

        if "label" not in dataset.column_names:
            return

        labels = dataset["label"]

        class_counts = {}

        for label in labels:

            class_counts[label] = (
                class_counts.get(label, 0) + 1
            )

        missing_classes = [
            label
            for label in range(self.num_classes)
            if class_counts.get(label, 0) == 0
        ]

        if missing_classes:

            self._add_error(
                f"{split_name}: Classes with zero "
                f"samples: {missing_classes}"
            )

    # ==================================================================
    # Validation pipeline
    # ==================================================================

    def validate(self) -> bool:
        """
        Execute the complete validation process.

        Returns:
            bool: True if validation passes.

        Raises:
            CustomException: If an unexpected exception occurs.
        """

        logger.info(
            "=================================================="
        )

        logger.info(
            "Starting Banking77 data validation."
        )

        logger.info(
            "=================================================="
        )

        try:

            # ----------------------------------------------------------
            # 1. Directory validation
            # ----------------------------------------------------------

            self.validate_directories()

            # ----------------------------------------------------------
            # 2. Dataset file validation
            # ----------------------------------------------------------

            self.validate_dataset_files()

            # If the raw structure itself is invalid, there is
            # no point attempting to load the datasets.
            if self.errors:

                logger.error(
                    "Raw dataset structure validation failed."
                )

                return False

            # ----------------------------------------------------------
            # 3. Load datasets
            # ----------------------------------------------------------

            self.load_datasets()

            # ----------------------------------------------------------
            # 4. Schema
            # ----------------------------------------------------------

            self.validate_columns(
                self.train_dataset,
                "train",
            )

            self.validate_columns(
                self.test_dataset,
                "test",
            )

            self.validate_dtypes(
                self.train_dataset,
                "train",
            )

            self.validate_dtypes(
                self.test_dataset,
                "test",
            )

            self.validate_schema_consistency()

            # ----------------------------------------------------------
            # 5. Row counts
            # ----------------------------------------------------------

            self.validate_row_counts()

            # ----------------------------------------------------------
            # 6. Missing values
            # ----------------------------------------------------------

            self.validate_missing_values(
                self.train_dataset,
                "train",
            )

            self.validate_missing_values(
                self.test_dataset,
                "test",
            )

            # ----------------------------------------------------------
            # 7. Text validation
            # ----------------------------------------------------------

            self.validate_text(
                self.train_dataset,
                "train",
            )

            self.validate_text(
                self.test_dataset,
                "test",
            )

            # ----------------------------------------------------------
            # 8. Duplicate validation
            # ----------------------------------------------------------

            self.validate_duplicates(
                self.train_dataset,
                "train",
            )

            self.validate_duplicates(
                self.test_dataset,
                "test",
            )

            # ----------------------------------------------------------
            # 9. Label validation
            # ----------------------------------------------------------

            self.validate_labels(
                self.train_dataset,
                "train",
            )

            self.validate_labels(
                self.test_dataset,
                "test",
            )

            # ----------------------------------------------------------
            # 10. Class distribution
            # ----------------------------------------------------------

            self.validate_class_distribution(
                self.train_dataset,
                "train",
            )

            self.validate_class_distribution(
                self.test_dataset,
                "test",
            )

            # ----------------------------------------------------------
            # 11. Leakage
            # ----------------------------------------------------------

            self.validate_train_test_leakage()

            # ----------------------------------------------------------
            # 12. Split ratio
            # ----------------------------------------------------------

            self.validate_split_ratio()

            # ----------------------------------------------------------
            # Final result
            # ----------------------------------------------------------

            if self.errors:

                logger.error(
                    "Banking77 validation FAILED with %d error(s).",
                    len(self.errors),
                )

                return False

            logger.info(
                "Banking77 validation PASSED."
            )

            if self.warnings:

                logger.warning(
                    "Validation completed with %d warning(s).",
                    len(self.warnings),
                )

            return True

        except CustomException:

            raise

        except Exception as error:

            raise CustomException(
                error,
                sys,
            )

    # ==================================================================
    # Validation report
    # ==================================================================

    def report(self) -> None:
        """Log a summary of the validation results."""

        logger.info(
            "---------------- Validation Report ----------------"
        )

        if self.train_dataset is not None:

            logger.info(
                "Train samples: %s",
                f"{len(self.train_dataset):,}",
            )

        if self.test_dataset is not None:

            logger.info(
                "Test samples: %s",
                f"{len(self.test_dataset):,}",
            )

        logger.info(
            "Validation errors: %d",
            len(self.errors),
        )

        logger.info(
            "Validation warnings: %d",
            len(self.warnings),
        )

        if self.errors:

            logger.error(
                "Validation status: FAILED"
            )

            for index, error in enumerate(
                self.errors,
                start=1,
            ):

                logger.error(
                    "Validation error %d: %s",
                    index,
                    error,
                )

        else:

            logger.info(
                "Validation status: PASSED"
            )

        if self.warnings:

            for index, warning in enumerate(
                self.warnings,
                start=1,
            ):

                logger.warning(
                    "Validation warning %d: %s",
                    index,
                    warning,
                )

        logger.info(
            "---------------------------------------------------"
        )


# ======================================================================
# Script entry point
# ======================================================================

if __name__ == "__main__":

    validator = Banking77Validator()

    validation_passed = validator.validate()

    validator.report()

    if not validation_passed:
        raise SystemExit(1)
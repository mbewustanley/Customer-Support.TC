# this file is built in 4 pieces 
"""
ClassDistributionAnalyser - which calculates train/test class counts and statistics
CSV artifact - reports/data/class_distribution.csv
PNG visualization - reports/data/class_distribution.png
Unit tests - so the analysis is reproducible and protected against regression

eventual flow ...
Validation
    ↓
Preprocessing
    ↓
Class Distribution Analysis
    ↓
Tokenization


NOTE:
- this analyzes processed data(not raw data) and should tell us what the model will recieve after invalid records have been removed.
- It will also calculate imbalance ratio and other statistics that will help us understand the data better and make decisions about how to handle it.


ClassDistributionAnalyzer class that accepts the processed huggingface dataset directly and return a structured dictionary of train, test and artifact data.
"""



# import necessary libraries
import sys

from pathlib import Path
from typing import Dict, Any

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import Dataset

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.exceptions import CustomException


logger = get_logger(__name__)

# Create analyser class
class ClassDistributionAnalyzer:
    """Analyse and report class distribution for the processed Banking77 dataset"""

    def __init__(
            self,
            config_path:str = "configs/data_config.yaml",
            report_dir:str = "reports/data"
            ):

        try:
            self.config = load_config(config_path)

            self.report_dir = Path(report_dir)
            self.report_dir.mkdir(
                parents=True,
                exist_ok=True
            )
            
            self.label_column = "label"
            self.num_classes = self.config["classification"]["num_classes"]

            self.csv_path = self.report_dir / "class_distribution.csv"
            self.plot_path = self.report_dir / "class_distribution.png"

            logger.info(
                f"Initialized ClassDistributionAnalyzer with report directory: {self.report_dir}"
            )

        except Exception as e:
            logger.error(f"Error initializing ClassDistributionAnalyzer: {e}")
            raise CustomException(e, sys) from e
    
        

    def _get_class_counts(self, dataset: Dataset) -> pd.Series:
        """Return the number of records belonging to each class."""

        try:
            labels = dataset[self.label_column]

            counts = (
                pd.Series(labels, dtype="int64")
                .value_counts()
                .reindex(
                    range(self.num_classes),
                    fill_value=0,
                )
                .sort_index()
            )

            counts.index.name = self.label_column
            counts.name = "count"

            return counts
        
        except Exception as e :
            logger.error(f"Error Calculating class counts: {e}")
            raise CustomException(e, sys) from e

    
    def build_distribution(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
    ) -> pd.DataFrame:
        """Build train/test class distribution dataframe."""

        try:
            logger.info("Building class distribution...")

            train_counts = self._get_class_counts(train_dataset)
            test_counts = self._get_class_counts(test_dataset)

            distribution = pd.DataFrame(
                {
                    "label": range(self.num_classes),
                    "train_count": train_counts.values,
                    "test_count": test_counts.values,
                }
            )

            train_total = len(train_dataset)
            test_total = len(test_dataset)

            distribution["train_percentage"] = (
                distribution["train_count"] / train_total * 100
            )

            distribution["test_percentage"] = (
                distribution["test_count"] / test_total * 100
            )

            logger.info(
                "Class distribution built for %d classes.",
                self.num_classes,
            )

            return distribution

        except Exception as e:
            logger.error(f"Error building class distribution: {e}")
            raise CustomException(e, sys) from e


    def calculate_statistics(self, distribution: pd.DataFrame) -> Dict[str, dict]:
        """Calculate distribution statistics for train and test datasets."""

        try:
            statistics = {}

            for split in ("train", "test"):
                counts = distribution[f"{split}_count"]

                min_count = int(counts.min())
                max_count = int(counts.max())

                statistics[split] = {
                    "num_samples": int(counts.sum()),
                    "num_classes": int((counts > 0).sum()),
                    "min_class_count": min_count,
                    "max_class_count": max_count,
                    "mean_class_count": float(counts.mean()),
                    "median_class_count": float(counts.median()),
                    "imbalance_ratio": (
                        float(max_count / min_count)
                        if min_count > 0
                        else float("inf")
                    ),
                }

            logger.info("Class distribution statistics calculated.")

            return statistics

        except Exception as e:
            logger.error(f"Error calculating class distribution statistics: {e}")
            raise CustomException(e, sys) from e

        

    def save_csv( self, distribution: pd.DataFrame,) -> Path:
       
        """Save class distribution as a CSV report."""

        try:
            distribution.to_csv(
                self.csv_path,
                index=False,
            )

            logger.info(
                "Class distribution CSV saved to %s",
                self.csv_path,
            )

            return self.csv_path

        except Exception as e:
            logger.error(
                f"Error saving class distribution CSV: {e}"
            )
            raise CustomException(e, sys) from e


    def save_plot(self, distribution: pd.DataFrame ) -> Path:
        """Save train/test class distribution plot."""

        try:
            plt.figure(figsize=(18, 7))

            x = distribution["label"]

            plt.bar(
                x - 0.2,
                distribution["train_count"],
                width=0.4,
                label="Train",
            )

            plt.bar(
                x + 0.2,
                distribution["test_count"],
                width=0.4,
                label="Test",
            )

            plt.xlabel("Class Label")
            plt.ylabel("Number of Samples")
            plt.title("Banking77 Class Distribution")
            plt.xticks(range(self.num_classes))
            plt.legend()
            plt.tight_layout()

            plt.savefig(
                self.plot_path,
                dpi=150,
            )

            plt.close()

            logger.info(
                "Class distribution plot saved to %s",
                self.plot_path,
            )

            return self.plot_path

        except Exception as e:
            logger.error(f"Error saving class distribution plot: {e}")
            
            raise CustomException(e, sys) from e

        

    def analyze( self, train_dataset: Dataset, test_dataset: Dataset ) -> Dict:

        """Run the complete class distribution analysis."""

        try:
            logger.info("Starting class distribution analysis.")

            distribution = self.build_distribution(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
            )

            statistics = self.calculate_statistics(
                distribution=distribution,
            )

            csv_path = self.save_csv(
                distribution=distribution,
            )

            plot_path = self.save_plot(
                distribution=distribution,
            )

            result = {
                "train": statistics["train"],
                "test": statistics["test"],
                "artifacts": {
                    "csv": str(csv_path),
                    "plot": str(plot_path),
                },
            }

            logger.info(
                "Class distribution analysis completed successfully."
            )

            return result

        except Exception as e:
            logger.error(
                f"Class distribution analysis failed: {e}"
            )
            raise CustomException(e, sys) from e
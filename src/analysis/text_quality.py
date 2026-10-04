from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset


class TextQualityAnalyzer:
    """
    Analyze text quality for a Hugging Face Dataset.

    Expected dataset columns:
        - text
        - label

    The analyzer does not modify the input dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        report_dir: str = "reports/data/text_quality",
        min_text_length: int = 3,
        max_text_length: int = 512,
    ) -> None:
        self.dataset = dataset
        self.report_dir = Path(report_dir)
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length

        self._validate_dataset()

        self.report_dir.mkdir(parents=True, exist_ok=True)

    def _validate_dataset(self) -> None:
        """Validate the minimum dataset contract."""
        required_columns = {"text", "label"}
        actual_columns = set(self.dataset.column_names)

        missing_columns = required_columns - actual_columns

        if missing_columns:
            raise ValueError(
                f"Dataset is missing required columns: {sorted(missing_columns)}"
            )

    def _to_dataframe(self) -> pd.DataFrame:
        """Convert the Hugging Face Dataset to a DataFrame."""
        return self.dataset.to_pandas()

    def calculate_statistics(self) -> Dict[str, Any]:
        """
        Calculate text quality statistics.

        Returns:
            Dictionary containing dataset-level quality metrics.
        """
        df = self._to_dataframe()

        text_series = df["text"]

        # Missing values
        missing_text_count = int(text_series.isna().sum())

        # Safely convert non-null values to strings for text analysis.
        text_as_string = text_series.fillna("").astype(str)

        # Empty and whitespace-only values
        empty_text_count = int((text_as_string == "").sum())
        whitespace_only_count = int(
            text_as_string.str.strip().eq("").sum()
        )

        # Text lengths
        text_lengths = text_as_string.str.len()

        # Duplicate analysis
        duplicate_text_count = int(
            text_as_string.duplicated(keep=False).sum()
        )

        duplicate_record_count = int(
            df.duplicated(subset=["text", "label"], keep=False).sum()
        )

        # Same text appearing with multiple labels
        label_conflict_count = int(
            df.groupby("text", dropna=False)["label"]
            .nunique()
            .gt(1)
            .sum()
        )

        # Suspicious lengths
        too_short_count = int(
            text_lengths.lt(self.min_text_length).sum()
        )

        too_long_count = int(
            text_lengths.gt(self.max_text_length).sum()
        )

        statistics = {
            "total_samples": int(len(df)),
            "missing_text_count": missing_text_count,
            "empty_text_count": empty_text_count,
            "whitespace_only_count": whitespace_only_count,
            "duplicate_text_count": duplicate_text_count,
            "duplicate_record_count": duplicate_record_count,
            "label_conflict_count": label_conflict_count,
            "too_short_count": too_short_count,
            "too_long_count": too_long_count,
            "text_length": {
                "min": int(text_lengths.min()) if len(text_lengths) else 0,
                "max": int(text_lengths.max()) if len(text_lengths) else 0,
                "mean": float(text_lengths.mean()) if len(text_lengths) else 0.0,
                "median": float(text_lengths.median()) if len(text_lengths) else 0.0,
                "std": float(text_lengths.std()) if len(text_lengths) > 1 else 0.0,
            },
        }

        return statistics

    def build_quality_report(self) -> pd.DataFrame:
        """
        Build a row-level quality report.

        Returns:
            DataFrame containing text and quality indicators.
        """
        df = self._to_dataframe().copy()

        text_series = df["text"].fillna("").astype(str)

        df["text_length"] = text_series.str.len()
        df["is_missing"] = df["text"].isna()
        df["is_empty"] = text_series.eq("")
        df["is_whitespace_only"] = text_series.str.strip().eq("")
        df["is_too_short"] = df["text_length"].lt(self.min_text_length)
        df["is_too_long"] = df["text_length"].gt(self.max_text_length)
        df["is_duplicate_text"] = text_series.duplicated(keep=False)
        df["is_duplicate_record"] = df.duplicated(
            subset=["text", "label"],
            keep=False,
        )

        # Identify text values associated with more than one label.
        label_counts = df.groupby("text")["label"].transform("nunique")
        df["has_label_conflict"] = label_counts.gt(1)

        return df

    def save_csv(self, report: Optional[pd.DataFrame] = None) -> Path:
        """
        Save the row-level quality report as CSV.

        Returns:
            Path to the generated CSV.
        """
        if report is None:
            report = self.build_quality_report()

        output_path = self.report_dir / "text_quality.csv"
        report.to_csv(output_path, index=False)

        return output_path

    def save_plot(self) -> Path:
        """
        Save text-length distribution plot.

        Returns:
            Path to the generated PNG.
        """
        df = self._to_dataframe()

        text_lengths = (
            df["text"]
            .fillna("")
            .astype(str)
            .str.len()
        )

        output_path = self.report_dir / "text_length_distribution.png"

        plt.figure(figsize=(10, 6))
        plt.hist(text_lengths, bins=50)
        plt.xlabel("Text Length")
        plt.ylabel("Frequency")
        plt.title("Banking77 Text Length Distribution")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def save_statistics(
        self,
        statistics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save summary statistics as JSON.

        Returns:
            Path to the generated JSON file.
        """
        if statistics is None:
            statistics = self.calculate_statistics()

        output_path = self.report_dir / "statistics.json"

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(statistics, file, indent=4)

        return output_path

    def analyze(self) -> Dict[str, Any]:
        """
        Run the complete text quality analysis.

        Returns:
            Dictionary containing statistics and artifact paths.
        """
        statistics = self.calculate_statistics()
        report = self.build_quality_report()

        csv_path = self.save_csv(report)
        plot_path = self.save_plot()
        statistics_path = self.save_statistics(statistics)

        return {
            "statistics": statistics,
            "artifacts": {
                "csv": str(csv_path),
                "plot": str(plot_path),
                "statistics": str(statistics_path),
            },
        }
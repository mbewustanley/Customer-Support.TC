from pathlib import Path

from datasets import load_from_disk

from src.analysis.class_distribution import ClassDistributionAnalyzer
from src.analysis.text_quality import TextQualityAnalyzer


def main() -> None:
    processed_dir = Path("data/processed")

    train_dataset = load_from_disk(processed_dir / "train")
    test_dataset = load_from_disk(processed_dir / "test")

    
    # Class distribution analysis
    class_distribution_analyzer = ClassDistributionAnalyzer(
        config_path="configs/data_config.yaml",
        report_dir="reports/data/class_distribution",
    )

    class_distribution_analyzer.analyze(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    
    # Text quality analysis
    text_quality_analyzer = TextQualityAnalyzer(
        dataset=train_dataset,
        report_dir="reports/data/text_quality",
    )

    text_quality_analyzer.analyze()


if __name__ == "__main__":
    main()
# run pytest tests/test_class_distribution.py -v


import pandas as pd
from datasets import Dataset

from src.analysis.class_distribution import ClassDistributionAnalyzer


def create_analyzer(tmp_path):
    return ClassDistributionAnalyzer(
        config_path="configs/data_config.yaml",
        report_dir=str(tmp_path / "reports"),
    )


def create_dataset(labels):
    return Dataset.from_dict(
        {
            "text": [f"sample {i}" for i in range(len(labels))],
            "label": labels,
        }
    )


def test_build_distribution(tmp_path):
    analyzer = create_analyzer(tmp_path)

    train_dataset = create_dataset([0, 0, 1, 1, 1, 2])
    test_dataset = create_dataset([0, 1, 2, 2])

    distribution = analyzer.build_distribution(
        train_dataset,
        test_dataset,
    )

    assert isinstance(distribution, pd.DataFrame)
    assert len(distribution) == 77

    assert distribution.loc[0, "train_count"] == 2
    assert distribution.loc[1, "train_count"] == 3
    assert distribution.loc[2, "train_count"] == 1

    assert distribution.loc[0, "test_count"] == 1
    assert distribution.loc[1, "test_count"] == 1
    assert distribution.loc[2, "test_count"] == 2


def test_calculate_statistics(tmp_path):
    analyzer = create_analyzer(tmp_path)

    train_dataset = create_dataset([0, 0, 1, 1, 1, 2])
    test_dataset = create_dataset([0, 1, 2, 2])

    distribution = analyzer.build_distribution(
        train_dataset,
        test_dataset,
    )

    statistics = analyzer.calculate_statistics(distribution)

    assert statistics["train"]["num_samples"] == 6
    assert statistics["test"]["num_samples"] == 4

    assert statistics["train"]["num_classes"] == 3
    assert statistics["test"]["num_classes"] == 3

    assert statistics["train"]["min_class_count"] == 0
    assert statistics["train"]["max_class_count"] == 3


def test_save_csv(tmp_path):
    analyzer = create_analyzer(tmp_path)

    train_dataset = create_dataset([0, 1, 2])
    test_dataset = create_dataset([0, 1, 2])

    distribution = analyzer.build_distribution(
        train_dataset,
        test_dataset,
    )

    csv_path = analyzer.save_csv(distribution)

    assert csv_path.exists()
    assert csv_path.name == "class_distribution.csv"


def test_save_plot(tmp_path):
    analyzer = create_analyzer(tmp_path)

    train_dataset = create_dataset([0, 1, 2])
    test_dataset = create_dataset([0, 1, 2])

    distribution = analyzer.build_distribution(
        train_dataset,
        test_dataset,
    )

    plot_path = analyzer.save_plot(distribution)

    assert plot_path.exists()
    assert plot_path.name == "class_distribution.png"


def test_full_analysis(tmp_path):
    analyzer = create_analyzer(tmp_path)

    train_dataset = create_dataset([0, 0, 1, 1, 1, 2])
    test_dataset = create_dataset([0, 1, 2, 2])

    result = analyzer.analyze(
        train_dataset,
        test_dataset,
    )

    assert result["train"]["num_samples"] == 6
    assert result["test"]["num_samples"] == 4

    assert "csv" in result["artifacts"]
    assert "plot" in result["artifacts"]

    assert (tmp_path / "reports" / "class_distribution.csv").exists()
    assert (tmp_path / "reports" / "class_distribution.png").exists()
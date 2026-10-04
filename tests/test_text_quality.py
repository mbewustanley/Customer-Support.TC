import json

import pytest
from datasets import Dataset

from src.analysis.text_quality import TextQualityAnalyzer


@pytest.fixture
def sample_dataset():
    return Dataset.from_dict(
        {
            "text": [
                "I need help with my card",
                "How do I transfer money?",
                "How do I transfer money?",
                "   ",
                "",
                "Short",
                "I need help with my card",
            ],
            "label": [
                0,
                1,
                1,
                2,
                3,
                4,
                5,
            ],
        }
    )


@pytest.fixture
def analyzer(sample_dataset, tmp_path):
    return TextQualityAnalyzer(
        dataset=sample_dataset,
        report_dir=str(tmp_path),
    )


def test_calculate_statistics(analyzer):
    statistics = analyzer.calculate_statistics()

    assert statistics["total_samples"] == 7
    assert statistics["empty_text_count"] == 1
    assert statistics["whitespace_only_count"] == 2
    assert statistics["duplicate_text_count"] >= 2
    assert statistics["label_conflict_count"] == 1


def test_build_quality_report(analyzer):
    report = analyzer.build_quality_report()

    assert len(report) == 7

    expected_columns = {
        "text",
        "label",
        "text_length",
        "is_missing",
        "is_empty",
        "is_whitespace_only",
        "is_too_short",
        "is_too_long",
        "is_duplicate_text",
        "is_duplicate_record",
        "has_label_conflict",
    }

    assert expected_columns.issubset(set(report.columns))


def test_save_csv(analyzer):
    report = analyzer.build_quality_report()

    output_path = analyzer.save_csv(report)

    assert output_path.exists()
    assert output_path.suffix == ".csv"


def test_save_plot(analyzer):
    output_path = analyzer.save_plot()

    assert output_path.exists()
    assert output_path.suffix == ".png"


def test_save_statistics(analyzer):
    statistics = analyzer.calculate_statistics()

    output_path = analyzer.save_statistics(statistics)

    assert output_path.exists()
    assert output_path.suffix == ".json"

    with output_path.open("r", encoding="utf-8") as file:
        saved_statistics = json.load(file)

    assert saved_statistics["total_samples"] == 7


def test_full_analysis(analyzer):
    result = analyzer.analyze()

    assert "statistics" in result
    assert "artifacts" in result

    assert "csv" in result["artifacts"]
    assert "plot" in result["artifacts"]
    assert "statistics" in result["artifacts"]

    for path in result["artifacts"].values():
        assert path


def test_missing_required_column():
    dataset = Dataset.from_dict(
        {
            "text": ["hello", "world"],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        TextQualityAnalyzer(dataset=dataset)
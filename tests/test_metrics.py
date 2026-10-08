import pytest

from src.evaluation.metrics import ClassificationMetrics


def test_perfect_predictions():
    predictions = [0, 1, 2, 3]
    labels = [0, 1, 2, 3]

    metrics = ClassificationMetrics.compute(predictions, labels)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_completely_incorrect_predictions():
    predictions = [1, 2, 0]
    labels = [0, 1, 2]

    metrics = ClassificationMetrics.compute(predictions, labels)

    assert metrics["accuracy"] == 0.0
    assert metrics["macro_precision"] == 0.0
    assert metrics["macro_recall"] == 0.0
    assert metrics["macro_f1"] == 0.0


def test_multiclass_predictions():
    predictions = [0, 1, 1, 2, 2, 2]
    labels = [0, 1, 2, 2, 2, 1]

    metrics = ClassificationMetrics.compute(predictions, labels)

    assert set(metrics.keys()) == {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    }

    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_all_77_classes():
    labels = list(range(77))
    predictions = list(range(77))

    metrics = ClassificationMetrics.compute(predictions, labels)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_mismatched_lengths_raise_error():
    predictions = [0, 1, 2]
    labels = [0, 1]

    with pytest.raises(ValueError):
        ClassificationMetrics.compute(predictions, labels)


def test_empty_inputs_raise_error():
    with pytest.raises(ValueError):
        ClassificationMetrics.compute([], [])


def test_metrics_return_floats():
    predictions = [0, 1, 2]
    labels = [0, 1, 2]

    metrics = ClassificationMetrics.compute(predictions, labels)

    for value in metrics.values():
        assert isinstance(value, float)
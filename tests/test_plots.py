from pathlib import Path

import pytest

from src.evaluation.plots import TrainingPlotter


@pytest.fixture
def training_history():
    return [
        {
            "epoch": 1,
            "train_loss": 1.20,
            "accuracy": 0.70,
            "macro_precision": 0.68,
            "macro_recall": 0.67,
            "macro_f1": 0.67,
        },
        {
            "epoch": 2,
            "train_loss": 0.80,
            "accuracy": 0.82,
            "macro_precision": 0.80,
            "macro_recall": 0.79,
            "macro_f1": 0.79,
        },
        {
            "epoch": 3,
            "train_loss": 0.50,
            "accuracy": 0.88,
            "macro_precision": 0.87,
            "macro_recall": 0.86,
            "macro_f1": 0.86,
        },
    ]


def test_plotter_initialization(
    training_history,
    tmp_path,
):

    plotter = TrainingPlotter(
        history=training_history,
        output_dir=str(tmp_path),
    )

    assert plotter.history == training_history
    assert plotter.output_dir == tmp_path
    assert tmp_path.exists()


def test_empty_history_rejected(tmp_path):

    with pytest.raises(ValueError):

        TrainingPlotter(
            history=[],
            output_dir=str(tmp_path),
        )


def test_invalid_history_type_rejected(tmp_path):

    with pytest.raises(TypeError):

        TrainingPlotter(
            history={},
            output_dir=str(tmp_path),
        )


def test_get_epochs(
    training_history,
    tmp_path,
):

    plotter = TrainingPlotter(
        history=training_history,
        output_dir=str(tmp_path),
    )

    epochs = plotter._get_epochs()

    assert epochs == [1, 2, 3]


def test_get_metric_values(
    training_history,
    tmp_path,
):

    plotter = TrainingPlotter(
        history=training_history,
        output_dir=str(tmp_path),
    )

    values = plotter._get_metric_values(
        "macro_f1"
    )

    assert values == [
        0.67,
        0.79,
        0.86,
    ]


def test_missing_metric_rejected(
    training_history,
    tmp_path,
):

    plotter = TrainingPlotter(
        history=training_history,
        output_dir=str(tmp_path),
    )

    with pytest.raises(KeyError):

        plotter._get_metric_values(
            "roc_auc"
        )


def test_plot_training_loss(
    training_history,
    tmp_path,
):

    plotter = TrainingPlotter(
        history=training_history,
        output_dir=str(tmp_path),
    )

    output_path = (
        plotter.plot_training_loss()
    )

    assert output_path.exists()
    assert output_path.name == "training_loss.png"
    assert output_path.stat().st_size > 0


def test_plot_metric(
    training_history,
    tmp_path,
):

    plotter = TrainingPlotter(
        history=training_history,
        output_dir=str(tmp_path),
    )

    output_path = plotter.plot_metric(
        "macro_f1"
    )

    assert output_path.exists()
    assert output_path.name == "macro_f1.png"
    assert output_path.stat().st_size > 0


def test_plot_all_metrics(
    training_history,
    tmp_path,
):

    plotter = TrainingPlotter(
        history=training_history,
        output_dir=str(tmp_path),
    )

    output_path = (
        plotter.plot_all_metrics(
            [
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
            ]
        )
    )

    assert output_path.exists()
    assert (
        output_path.name
        == "evaluation_metrics.png"
    )
    assert output_path.stat().st_size > 0


def test_generate_all(
    training_history,
    tmp_path,
):

    metrics = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]

    plotter = TrainingPlotter(
        history=training_history,
        output_dir=str(tmp_path),
    )

    plots = plotter.generate_all(
        metrics
    )

    assert "training_loss" in plots
    assert "accuracy" in plots
    assert "macro_precision" in plots
    assert "macro_recall" in plots
    assert "macro_f1" in plots
    assert "evaluation_metrics" in plots

    for path in plots.values():
        assert isinstance(path, Path)
        assert path.exists()
        assert path.stat().st_size > 0
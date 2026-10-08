"""
python -m src.training.train


Responsibilities:
1. load config
2. tokenize processed data
3. wrap datasets
4. create dataloaders
5. create classifier
6. create trainer
7. trainer.train()
8. trainingplotter.generate_all()
return training results



                 Banking77
                     │
                     ▼
              Data Validation
                     │
                     ▼
               Preprocessing
                     │
                     ▼
             Class Distribution
                     │
                     ▼
                Tokenization
                     │
                     ▼
              PyTorch Dataset
                     │
                     ▼
                DataLoader
                     │
                     ▼
            DistilBERT Classifier
                     │
                     ▼
                  Trainer
              ┌──────┴──────┐
              │             │
          Evaluation    Checkpoint
              │             │
              ▼             ▼
          Metrics      best_model.pt
              │
              ▼
        Training History
              │
              ▼
        TrainingPlotter
              │
              ▼
        artifacts/reports
"""

from pathlib import Path
from typing import Any, Dict, Tuple
from datasets import load_from_disk
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data.dataset import Banking77Dataset
from src.evaluation.plots import TrainingPlotter
from src.models.classifier import Banking77Classifier
from src.tokenization.tokenizer import Banking77Tokenizer
from src.training.trainer import Trainer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger


logger = get_logger(__name__)


def create_dataloaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    """
    Load tokenized Banking77 datasets and create
    PyTorch DataLoaders.
    """

    training_config = config["training"]

    tokenized_data_dir = Path(
        training_config.get(
            "tokenized_data_dir",
            "data/tokenized",
        )
    )

    train_path = tokenized_data_dir / "train"
    test_path = tokenized_data_dir / "test"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Tokenized train dataset not found: {train_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Tokenized test dataset not found: {test_path}"
        )

    train_dataset = load_from_disk(str(train_path))
    test_dataset = load_from_disk(str(test_path))


    validation_ratio = training_config.get("validation_ratio", 0.1)
    seed = training_config.get("seed", 42)

    batch_size = training_config["batch_size"]
    eval_batch = training_config["eval_batch"]
    num_workers = training_config.get(
        "num_workers",
        0,
    )

    train_dataset, validation_dataset = split_training_dataset(
        train_dataset, validation_ratio=validation_ratio, seed=seed
    )

    train_dataset = Banking77Dataset(train_dataset)
    validation_dataset = Banking77Dataset(validation_dataset)
    test_dataset = Banking77Dataset(test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=eval_batch,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch,
        shuffle=False,
        num_workers=num_workers
    )

    logger.info(
        "Created train DataLoader with %d samples.",
        len(train_dataset),
    )

    logger.info(
        "Created validation DataLoader with %d samples.",
        len(validation_dataset)
    )

    logger.info(
            "Created Test DataLoader with %d samples.",
            len(test_dataset)
        )

    return train_loader, validation_loader, test_loader


def create_model(
    config: Dict[str, Any],
) -> Banking77Classifier:
    """Create the configured Banking77 classifier."""

    training_config = config["training"]

    model = Banking77Classifier(
        model_name=training_config["model_name"],
        num_labels=training_config["num_labels"],
    )

    logger.info(
        "Created model: %s",
        training_config["model_name"],
    )

    return model


def ensure_tokenized_data(
    config: Dict[str, Any],
) -> None:
    """
    Ensure tokenized datasets exist.

    Existing tokenized data is reused. If it does not
    exist, the tokenizer creates it.
    """

    training_config = config["training"]

    tokenized_data_dir = Path(
        training_config.get(
            "tokenized_data_dir",
            "data/tokenized",
        )
    )

    train_path = tokenized_data_dir / "train"
    test_path = tokenized_data_dir / "test"

    if train_path.exists() and test_path.exists():

        logger.info(
            "Tokenized datasets already exist. "
            "Skipping tokenization."
        )

        return

    logger.info(
        "Tokenized datasets not found. "
        "Starting tokenization."
    )

    tokenizer = Banking77Tokenizer(
        model_name=training_config["model_name"],
        processed_data_dir="data/processed",
        tokenized_data_dir=str(
            tokenized_data_dir
        ),
        max_length=training_config.get(
            "max_length",
            128,
        ),
    )

    tokenizer.run()


def split_training_dataset(
    dataset,
    validation_ratio: float,
    seed: int,
):
    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    indices = list(range(len(dataset)))
    labels = dataset["label"]

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_ratio,
        random_state=seed,
        stratify=labels,
    )

    train_dataset = dataset.select(train_indices)
    validation_dataset = dataset.select(validation_indices)

    logger.info(
        "Split training dataset: %d train / %d validation.",
        len(train_dataset),
        len(validation_dataset)
    )

    return train_dataset, validation_dataset



def run_training(
    config_path: str = "configs/training_config.yaml",
) -> Dict[str, Any]:
    """
    Run the complete model training pipeline.
    """

    logger.info(
        "Loading training configuration from %s",
        config_path,
    )

    config = load_config(config_path)

    ensure_tokenized_data(config)

    train_loader, validation_loader, test_loader = (
        create_dataloaders(config)
    )

    model = create_model(config)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=validation_loader,
        config=config,
    )

    results = trainer.train()


    # Evaluation on the untouched test set
    test_metrics = trainer.evaluate(test_loader)

    results["test_metrics"] = test_metrics


    plotter = TrainingPlotter(
        history=results["history"],
        output_dir=config["training"].get(
            "report_dir",
            "artifacts/reports",
        ),
    )

    plots = plotter.generate_all(
        config["evaluation"]["metrics"]
    )

    results["plots"] = {
        name: str(path)
        for name, path in plots.items()
    }

    logger.info(
        "Training pipeline completed successfully."
    )

    return results


if __name__ == "__main__":
    run_training()
# Banking77 Customer Support Classification — End-to-End MLOps

A production-oriented NLP and MLOps project that classifies customer support messages into 77 banking-related intents using **PyTorch, Hugging Face Transformers, and DistilBERT**.

The goal is to build more than a model that achieves good accuracy. This project is being developed as a reproducible, testable, and maintainable machine learning pipeline, with experiment tracking, model versioning, cloud artifact storage, and deployment planned as subsequent stages.

---

## 1. Project Objectives

This project aims to demonstrate practical machine learning engineering and MLOps skills through a complete NLP workflow.

The main objectives are to:

- Build a reliable data validation and preprocessing pipeline.
- Analyze class distributions and dataset quality.
- Tokenize text using a pretrained Hugging Face tokenizer.
- Train a transformer-based text classifier using PyTorch.
- Separate training, validation, and final test data correctly.
- Track training and evaluation metrics.
- Save model checkpoints and evaluation reports.
- Test individual components and the end-to-end training workflow.
- Extend the pipeline with experiment tracking, data versioning, cloud storage, inference serving, and deployment.

### Technology stack

| Area                                 | Technology                |
| ------------------------------------ | ------------------------- |
| Language                             | Python 3.10               |
| Deep learning                        | PyTorch                   |
| NLP and pretrained models            | Hugging Face Transformers |
| Dataset management                   | Hugging Face Datasets     |
| Data processing                      | Pandas, NumPy             |
| Machine learning utilities           | Scikit-learn              |
| Configuration                        | YAML                      |
| Testing                              | Pytest                    |
| Visualization                        | Matplotlib                |
| Version control                      | Git and GitHub            |
| Planned experiment tracking          | MLflow                    |
| Planned data and pipeline versioning | DVC                       |
| Planned model serving                | FastAPI                   |
| Planned containerization             | Docker                    |
| Planned cloud storage                | AWS S3                    |

---

## 2. Dataset

**Dataset:** [Banking77 — Hugging Face](https://huggingface.co/datasets/PolyAI/banking77)

Banking77 is a banking customer-support intent classification dataset containing customer messages associated with 77 distinct intents.

### Dataset summary

| Property                              |                  Value |
| ------------------------------------- | ---------------------: |
| Training examples in original dataset |                 10,003 |
| Official test examples                |                  3,080 |
| Total examples                        |                 13,083 |
| Number of intent classes              |                     77 |
| Input column                          |                 `text` |
| Target column                         |                `label` |
| Label representation                  | Zero-based integer IDs |

An example input is:

> I am still waiting on my card?

The model learns to map customer messages to one of the 77 intent classes.

The dataset is loaded from the Hugging Face Hub and stored locally in Hugging Face's Arrow-based dataset format for subsequent pipeline stages.

---

## 3. Project Architecture

The project follows a modular structure, separating data processing, analysis, tokenization, model training, evaluation, and supporting utilities.

```text
Customer-Support.TC/
│
├── configs/
│   ├── data_config.yaml
│   └── training_config.yaml
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   └── test/
│   ├── processed/
│   └── tokenized/
│
├── src/
│   ├── analysis/
│   │   └── class_distribution.py
│   │
│   ├── data/
│   │   ├── validation.py
│   │   └── dataset.py
│   │
│   ├── preprocessing/
│   │   └── preprocessing.py
│   │
│   ├── tokenization/
│   │   └── tokenizer.py
│   │
│   ├── models/
│   │   └── classifier.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   └── train.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── plots.py
│   │
│   └── utils/
│       ├── logger.py
│       └── exceptions.py
│
├── tests/
│   ├── test_validation.py
│   ├── test_preprocessing.py
│   ├── test_class_distribution.py
│   ├── test_train.py
│   └── ...
│
├── artifacts/
│   ├── checkpoints/
│   ├── models/
│   └── reports/
│
├── requirements.txt
├── pyproject.toml
├── dvc.yaml
└── README.md
```

_Note: This tree documents the known project structure and logical artifact locations. It is not a guarantee that every directory or file currently exists in the repository, or that all generated data and artifacts are committed to Git._

---

## 4. Configuration Management

Configuration is kept outside the main implementation so that important parameters can be changed without modifying the pipeline's code.

### Data configuration

`configs/data_config.yaml` contains dataset-related settings, including:

- Hugging Face dataset identifier.
- Expected column names and data types.
- Expected dataset sizes.
- Number of classes and label-indexing convention.
- Expected train/test split proportions and tolerance.
- Text validation rules.
- Preprocessing-related settings.

The dataset configuration is used by validation and preprocessing components to check whether the data conforms to the project's expectations.

### Training configuration

`configs/training_config.yaml` defines the principal training parameters:

```yaml
training:
  model_name: distilbert-base-uncased
  num_labels: 77
  batch_size: 16
  eval_batch: 16
  max_length: 128
  num_workers: 0
  tokenized_data_dir: data/tokenized
  epochs: 3
  learning_rate: 0.00002
  weight_decay: 0.01
  warmup_ratio: 0.1
  seed: 42
  validation_ratio: 0.1
  device: auto
  output_dir: artifacts/models
  checkpoint_dir: artifacts/checkpoints
  report_dir: artifacts/reports

evaluation:
  primary_metric: macro_f1
  metrics:
    - accuracy
    - macro_precision
    - macro_recall
    - macro_f1
```

These settings establish a reproducible baseline configuration and make it easier to run future experiments with different hyperparameters.

---

## 5. Implemented Pipeline Components

### 5.1 Data validation

**Module:** `src/data/validation.py`

The `Banking77Validator` validates the raw dataset before preprocessing.

Its checks include:

- Required raw-data directories and files.
- Expected columns and data types.
- Schema consistency.
- Expected row counts.
- Missing values.
- Empty or whitespace-only text.
- Duplicate records and duplicate text warnings.
- Label type, range, and class coverage.
- Missing expected classes.
- Potential train/test text leakage.
- Training/test split proportions.

The validator reports errors and warnings, helping identify data-quality problems before they affect model training.

**Testing status:** 15 tests passed.

### 5.2 Text preprocessing

**Module:** `src/preprocessing/preprocessing.py`

The `Banking77Preprocessor` transforms validated data into a cleaner form suitable for tokenization.

Implemented behavior includes:

- Loading the configured dataset.
- Handling text and label columns.
- Cleaning leading and trailing whitespace.
- Removing records with missing or empty text.
- Converting labels to integers where valid.
- Rejecting negative and out-of-range labels.
- Saving processed data to the configured output directory.

**Testing status:** 11 tests passed.

### 5.3 Class distribution analysis

**Module:** `src/analysis/class_distribution.py`

The class distribution analyzer examines the frequency of each intent class.

It supports:

- Building class-frequency distributions.
- Calculating distribution statistics.
- Saving results as CSV.
- Generating distribution plots.
- Running the full analysis workflow.

This stage provides visibility into class balance and helps guide later evaluation and error analysis.

**Testing status:** 5 tests passed, covering distribution construction, statistics, CSV output, plot output, and the complete analysis workflow.

### 5.4 Tokenization

**Module:** `src/tokenization/tokenizer.py`

The tokenization stage uses the Hugging Face tokenizer associated with `distilbert-base-uncased`.

Its purpose is to convert customer messages into model-ready token representations, including token IDs and attention masks, using the configured maximum sequence length.

Tokenized datasets are saved under `data/tokenized` and reused when already available, avoiding unnecessary repeated tokenization.

The latest training run confirmed that the existing tokenized datasets were detected and reused successfully.

### 5.5 PyTorch dataset wrapper

**Module:** `src/data/dataset.py`

The `Banking77Dataset` adapts tokenized Hugging Face data for PyTorch training.

It provides the individual samples required by PyTorch `DataLoader` instances and supports batching of tokenized inputs and classification labels.

The latest run successfully created dataset wrappers for the training, validation, and test partitions.

### 5.6 Transformer classifier

**Module:** `src/models/classifier.py`

The classifier uses the pretrained `distilbert-base-uncased` model with a sequence-classification head configured for 77 labels.

The pretrained language representation provides a starting point, while the classification head is trained to predict Banking77 intents.

The model was successfully initialized and executed on a Google Colab NVIDIA T4 GPU.

**Important:** Hugging Face reported that some pretrained masked-language-modeling head parameters were unused and that new sequence-classification head parameters were initialized. This is expected when loading a base DistilBERT checkpoint for a different task.

### 5.7 Training orchestration

**Modules:**

- `src/training/trainer.py`
- `src/training/train.py`

The training implementation coordinates model training, validation, checkpoint selection, and final test evaluation.

Implemented behavior includes:

- Loading training configuration.
- Creating the train, validation, and test data loaders.
- Initializing the configured model.
- Selecting the available training device.
- Training over multiple epochs.
- Calculating validation metrics after each epoch.
- Saving the best-performing checkpoint based on validation macro-F1.
- Evaluating the selected model on the official test partition.
- Generating evaluation reports and plots.

The pipeline also includes a reproducible train/validation split using a configured random seed and stratification by label.

### 5.8 Evaluation metrics

**Module:** `src/evaluation/metrics.py`

The evaluation component calculates:

- Accuracy.
- Macro precision.
- Macro recall.
- Macro-F1.

Macro-averaged metrics give each class equal weight, making them useful for evaluating performance across the 77 intents rather than allowing frequently occurring classes to dominate the overall score.

### 5.9 Training and evaluation plots

**Module:** `src/evaluation/plots.py`

The plotting component generates visual reports for model performance.

The latest successful run generated:

- `artifacts/reports/training_loss.png`
- `artifacts/reports/accuracy.png`
- `artifacts/reports/macro_precision.png`
- `artifacts/reports/macro_recall.png`
- `artifacts/reports/macro_f1.png`
- `artifacts/reports/evaluation_metrics.png`

These visualizations help assess training progress and compare evaluation metrics.

### 5.10 Logging and exception handling

**Modules:**

- `src/utils/logger.py`
- `src/utils/exceptions.py`

The project has centralized logging and a custom exception mechanism to make pipeline failures easier to trace and diagnose.

These utilities are intended to keep error reporting consistent across modules rather than relying on unrelated ad hoc error handling.

---

## 6. Training, Validation, and Test Strategy

An important improvement to the training pipeline was separating validation from final test evaluation.

The original training dataset is split into a training subset and a validation subset. The official test set remains held out until final evaluation.

### Actual split sizes

| Partition     | Number of examples | Purpose                                       |
| ------------- | -----------------: | --------------------------------------------- |
| Training      |              9,002 | Learn model parameters                        |
| Validation    |              1,001 | Compare epochs and select the best checkpoint |
| Official test |              3,080 | Final held-out evaluation                     |
| **Total**     |         **13,083** |                                               |

The 10,003 original training examples are split using the configured 10% validation ratio, random seed, and label stratification.

The validation set is used to select the best checkpoint based on macro-F1. The official test set is evaluated separately after training.

This prevents the official test set from being used for routine epoch-by-epoch model selection.

---

## 7. Baseline Training Results

The first successful end-to-end GPU training run used:

| Parameter                   | Value                      |
| --------------------------- | -------------------------- |
| Model                       | `distilbert-base-uncased`  |
| Number of classes           | 77                         |
| Epochs                      | 3                          |
| Batch size                  | 16                         |
| Learning rate               | 0.00002                    |
| Weight decay                | 0.01                       |
| Warmup ratio                | 0.1                        |
| Maximum sequence length     | 128                        |
| Random seed                 | 42                         |
| Validation selection metric | Macro-F1                   |
| Hardware                    | Google Colab NVIDIA T4 GPU |

### Validation results by epoch

| Epoch | Training loss | Accuracy | Macro precision | Macro recall | Macro-F1 |
| ----- | ------------: | -------: | --------------: | -----------: | -------: |
| 1     |        3.3675 |   0.7203 |          0.6986 |       0.6692 |   0.6430 |
| 2     |        1.4102 |   0.8501 |          0.8278 |       0.8204 |   0.8135 |
| 3     |        0.8403 |   0.8811 |          0.8677 |       0.8591 |   0.8585 |

Validation macro-F1 improved at each epoch, and the third epoch produced the best observed validation result.

### Final test results

| Metric          | Test result |
| --------------- | ----------: |
| Accuracy        |      0.8575 |
| Macro precision |      0.8643 |
| Macro recall    |      0.8575 |
| Macro-F1        |      0.8499 |

The final test macro-F1 was approximately **0.8499**, compared with the best validation macro-F1 of approximately **0.8585**.

These results establish the initial benchmark for future experiments. They are a baseline, not a claim that the model is production-ready.

---

## 8. Development Environment and Execution

### Local development

The project was developed on Windows using a Conda environment.

| Component         | Local environment   |
| ----------------- | ------------------- |
| Operating system  | Windows             |
| Python            | 3.10.21             |
| Conda environment | `pytorch-nlp-mlops` |
| PyTorch           | 2.14.1+cpu          |
| Transformers      | 5.18.0              |
| Datasets          | 5.0.1               |
| Pytest            | 9.1.1               |
| Training device   | CPU                 |

The local machine was used for development, data-pipeline work, and testing. CUDA was not available in the local environment.

### GPU training environment

The end-to-end training run was executed in Google Colab using an NVIDIA T4 GPU.

The training logs confirmed CUDA availability, successful model initialization, training over three epochs, checkpoint saving, test evaluation, and report generation.

### Environment setup

Clone the repository and create the project environment:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Customer-Support.TC

conda create -n pytorch-nlp-mlops python=3.10
conda activate pytorch-nlp-mlops

pip install -r requirements.txt
```

Use the dependency versions and installation instructions maintained by the project when recreating the environment. GPU-enabled PyTorch installation should follow the requirements of the selected execution environment.

### Running tests

Run the test suite from the repository root:

```bash
pytest -v
```

The project's individual test modules have been used to validate the implemented components, and the full local test suite was reported green during development.

### Running the training pipeline

The training orchestration entry point is:

```text
src/training/train.py
```

Run the project using the entry-point invocation currently supported by the repository. For example, if the module is configured for package execution:

```bash
python -m src.training.train
```

The training pipeline loads its configuration, prepares data loaders, initializes the model, trains, selects a checkpoint using validation metrics, evaluates the test set, and generates plots.

---

## 9. Testing and Quality Assurance

Testing has been incorporated throughout development rather than postponed until the end.

The implemented tests cover data validation, preprocessing, class distribution analysis, training orchestration, and other individual components.

| Component                   | Recorded test result |
| --------------------------- | -------------------: |
| Data validation             |            15 passed |
| Preprocessing               |            11 passed |
| Class distribution analysis |             5 passed |
| Training orchestration      |             9 passed |
| Evaluation plots            |            10 passed |

These results are historical results from the relevant test runs; they should not be interpreted as proof that every current test passes after any subsequent code changes. Run the complete suite again before a release.

Additional quality checks to implement include:

- Testing checkpoint loading in a fresh process.
- Testing tokenizer/model compatibility.
- Testing inference on individual messages and batches.
- Verifying prediction output dimensions and valid class IDs.
- Verifying that test metrics correspond to the official 3,080-example test set.
- Testing failure cases for missing configuration, corrupt artifacts, and invalid inputs.

---

## 10. Version Control and Data Management

### Git and GitHub

The project has been pushed to GitHub.

Git is used to version the source code, tests, configuration files, and documentation. Large generated files and datasets should be managed according to a deliberate artifact-storage policy rather than committed indiscriminately.

### DVC

A `dvc.yaml` file exists in the project, and DVC has been explored as the mechanism for data and pipeline versioning.

However, a DVC remote was not configured at the time of the last recorded check: `dvc remote list` returned no configured remotes.

Therefore, **remote-backed DVC storage is not yet considered complete**. The next DVC step is to inspect the current pipeline definitions, confirm the stages and dependencies, configure an appropriate remote, and verify that data and artifacts can be pushed and retrieved.

---

## 11. Current Project Status

### Completed

- [x] Project structure and configuration files established.
- [x] Banking77 dataset selected and loaded.
- [x] Raw-data validation implemented and tested.
- [x] Text preprocessing implemented and tested.
- [x] Class distribution analysis implemented and tested.
- [x] Tokenization and PyTorch dataset integration implemented.
- [x] DistilBERT classifier configured for 77 intent classes.
- [x] Training and evaluation components implemented.
- [x] Reproducible stratified training/validation split implemented.
- [x] Best-checkpoint selection based on validation macro-F1 implemented.
- [x] Separate final test evaluation implemented.
- [x] End-to-end GPU training completed successfully.
- [x] Initial validation and test benchmark recorded.
- [x] Evaluation plots generated.
- [x] Source code pushed to GitHub.

### In progress or requiring verification

- [ ] Verify checkpoint contents and fresh-process loading.
- [ ] Ensure final test metrics and training history are persisted in machine-readable files.
- [ ] Confirm inference works using the saved model and matching tokenizer.
- [ ] Inspect and verify the current DVC pipeline stages.
- [ ] Configure and test a DVC remote.

### Planned

- [ ] Integrate MLflow experiment tracking.
- [ ] Log parameters, metrics, model artifacts, and evaluation reports.
- [ ] Introduce model versioning and a model registry.
- [ ] Configure durable cloud artifact storage, potentially AWS S3.
- [ ] Build a standalone inference module.
- [ ] Expose predictions through a FastAPI service.
- [ ] Containerize the application with Docker.
- [ ] Add continuous integration and automated quality checks.
- [ ] Deploy the service and implement operational monitoring.

---

## 12. Next Development Milestones

The next steps will be implemented in order to avoid introducing infrastructure before the underlying model lifecycle is reliable.

### Milestone 1 — Artifact persistence and verification

Persist training history and final test metrics, record checkpoint metadata, and confirm that the best model can be loaded and used independently of the training process.

### Milestone 2 — MLflow integration

Track training runs, configurations, metrics, plots, and model artifacts. Establish a reproducible record of each experiment and make it possible to compare future runs with the baseline.

### Milestone 3 — DVC and cloud storage

Complete DVC pipeline and remote configuration, then establish durable storage for datasets and model artifacts. Keep source-code versioning, dataset versioning, experiment metadata, and large artifact storage as distinct responsibilities.

### Milestone 4 — Inference and serving

Implement a standalone prediction interface, test it with unseen customer messages, and create a FastAPI endpoint for serving predictions.

### Milestone 5 — Deployment and monitoring

Containerize the inference service, automate tests and build checks, deploy it to a suitable environment, and introduce monitoring for model quality and service health.

---

## 13. Engineering Principles

The project is being developed around the following principles:

1. **Reproducibility:** Record configurations, seeds, data splits, and experiment results.
2. **Separation of concerns:** Keep data processing, modeling, training, evaluation, and serving in distinct modules.
3. **Testability:** Validate components independently and run automated tests throughout development.
4. **Leakage prevention:** Use validation for model selection and reserve the official test set for final evaluation.
5. **Traceability:** Connect model artifacts to their configurations and measured performance.
6. **Versioning:** Use Git for code, DVC for versioned data and pipeline artifacts, and MLflow for experiment and model lifecycle tracking.
7. **Deployment readiness:** Make model inference independent of the training workflow.
8. **Incremental engineering:** Establish a reliable baseline before introducing additional infrastructure.

---

## 14. Summary

This project has progressed from initial environment setup to a functioning, modular NLP training pipeline. The data validation, preprocessing, class distribution analysis, model training, evaluation, and plotting components have been implemented and tested, and a DistilBERT classifier has completed a GPU training run.

The initial benchmark is **85.75% test accuracy and 0.8499 test macro-F1** on the official Banking77 test set.

The immediate priority is to make the training results and checkpoint independently verifiable. MLflow integration, remote artifact storage, inference serving, and deployment will follow in stages.

The long-term goal is a documented, reproducible, end-to-end MLOps system that demonstrates practical machine learning engineering rather than model training alone.

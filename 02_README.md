# Enterprise NLP Classification MLOps Platform

A production-oriented NLP classification system built with **PyTorch + Hugging Face Transformers**, surrounded by a complete MLOps lifecycle.

The goal is not to build a toy deep-learning model. The goal is to demonstrate how an NLP model can be **developed, versioned, trained, evaluated, registered, deployed, monitored, and retrained** in a reproducible production-style environment.

---

## 1. Project Goal

Build an enterprise-style **Customer Support Ticket Classification** platform.

Given a customer support message such as:

> "My card was charged twice for the same transaction."

the system should return something like:

```json
{
  "prediction": "billing_issue",
  "confidence": 0.9472,
  "model_version": "3",
  "model_id": "customer-ticket-classifier"
}
```

The project will use a pretrained Hugging Face Transformer and fine-tune it with PyTorch.

The same MLOps infrastructure should eventually support:

- Different Transformer architectures
- Different datasets
- Full fine-tuning
- Parameter-efficient fine-tuning such as LoRA
- CPU inference
- GPU inference
- Local development
- Docker deployment
- AWS deployment
- Automated CI/CD
- Model monitoring
- Data/model drift detection
- Model retraining

---

# 2. Project Objectives

## Machine Learning

- Learn practical PyTorch for production ML engineering.
- Fine-tune a pretrained Hugging Face Transformer.
- Build reproducible training and evaluation pipelines.
- Handle tokenization and sequence processing correctly.
- Track experiments and hyperparameters.
- Compare model versions.
- Implement quality gates before deployment.
- Explore LoRA/PEFT after the baseline system works.

## MLOps

- Version datasets with DVC.
- Version code with Git.
- Track experiments with MLflow.
- Register models with MLflow Model Registry.
- Build reproducible training pipelines.
- Package inference with FastAPI.
- Containerize the application with Docker.
- Automate testing and deployment with GitHub Actions.
- Deploy to AWS.
- Monitor inference and model behavior.
- Detect data drift.
- Establish a retraining workflow.

## Engineering

The final project should demonstrate:

- Clean repository structure
- Configuration-driven execution
- Separation of concerns
- Unit/integration tests
- Reproducibility
- Structured logging
- Error handling
- API validation
- Model versioning
- CI/CD
- Production-oriented documentation

---

# 3. Target Architecture

```text
                         DATA SOURCES
                              |
                              v
                    +-------------------+
                    | Data Validation   |
                    | Schema / Quality  |
                    +---------+---------+
                              |
                              v
                            DVC
                     Dataset Versioning
                              |
                              v
                  +---------------------+
                  | Data Preprocessing  |
                  | Cleaning            |
                  | Label Encoding      |
                  | Train/Val/Test      |
                  +----------+----------+
                             |
                             v
                  Hugging Face Tokenizer
                             |
                             v
                  +---------------------+
                  |       PyTorch       |
                  |                     |
                  | AutoModelFor        |
                  | SequenceClassification
                  +----------+----------+
                             |
                             v
                        Fine-tuning
                             |
                  +----------+----------+
                  |                     |
                  v                     v
             Checkpoints           Evaluation
                  |                     |
                  +----------+----------+
                             |
                             v
                          MLflow
                    Experiment Tracking
                             |
                             v
                      Model Registry
                             |
                             v
                    +----------------+
                    | Quality Gates  |
                    +-------+--------+
                            |
                            v
                        Production
                            |
                   +--------+--------+
                   |                 |
                   v                 v
                FastAPI            Docker
                   |                 |
                   +--------+--------+
                            |
                            v
                           AWS
                            |
                            v
                     Monitoring
                            |
                +-----------+-----------+
                |                       |
                v                       v
          Data Drift              Service/Model
          Monitoring               Monitoring
```

CI/CD surrounds the lifecycle:

```text
                         GitHub
                           |
                           v
                    GitHub Actions
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
        Tests           Docker           Training
          |                |                |
          +----------------+----------------+
                           |
                           v
                        Deploy
```

---

# 4. Technology Stack

## Core ML

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Hugging Face Tokenizers
- Hugging Face Datasets where appropriate
- scikit-learn
- NumPy
- pandas

## MLOps

- MLflow
- DVC
- Git
- GitHub
- GitHub Actions

## Model Fine-tuning

Initial model:

```text
distilbert-base-uncased
```

The architecture should remain configurable so other models can be tested later.

Potential future models:

- BERT
- RoBERTa
- DistilBERT
- DeBERTa
- other compatible Hugging Face encoder models

## Parameter-Efficient Fine-Tuning

Later phase:

- PEFT
- LoRA
- QLoRA where hardware/software compatibility permits

## API

- FastAPI
- Pydantic
- Uvicorn

## Deployment

- Docker
- AWS ECR
- AWS EC2
- AWS S3
- AWS IAM
- CloudWatch where appropriate

## Monitoring

- Evidently
- Python logging
- API/service metrics

## Testing and Quality

- pytest
- Ruff or equivalent linting
- formatting/type checking as appropriate

---

# 5. High-Level Model Lifecycle

```text
Dataset
   |
   v
DVC version
   |
   v
Preprocessing
   |
   v
Tokenizer
   |
   v
Hugging Face pretrained model
   |
   v
PyTorch fine-tuning
   |
   v
Evaluation
   |
   v
Quality Gate
   |
   +---- FAIL ----> Reject
   |
   +---- PASS ----> MLflow
                         |
                         v
                   Model Registry
                         |
                         v
                      Staging
                         |
                         v
                     Production
                         |
                         v
                     Monitoring
                         |
                         v
                  Drift / Degradation
                         |
                         v
                     Retraining
```

---

# 6. Repository Structure

Target structure:

```text
pytorch-nlp-mlops/
|
+-- configs/
|   +-- base.yaml
|   +-- training.yaml
|   +-- model.yaml
|
+-- data/
|   +-- raw/
|   +-- processed/
|   +-- external/
|
+-- src/
|   +-- data/
|   |   +-- ingestion.py
|   |   +-- validation.py
|   |   +-- preprocessing.py
|   |
|   +-- features/
|   |   +-- tokenizer.py
|   |
|   +-- models/
|   |   +-- model.py
|   |   +-- factory.py
|   |
|   +-- training/
|   |   +-- train.py
|   |   +-- trainer.py
|   |   +-- checkpoint.py
|   |
|   +-- evaluation/
|   |   +-- evaluate.py
|   |   +-- metrics.py
|   |   +-- plots.py
|   |
|   +-- inference/
|       +-- predictor.py
|
+-- api/
|   +-- main.py
|   +-- schemas.py
|   +-- dependencies.py
|
+-- pipelines/
|   +-- training_pipeline.py
|   +-- evaluation_pipeline.py
|
+-- monitoring/
|   +-- drift.py
|
+-- tests/
|   +-- test_data.py
|   +-- test_model.py
|   +-- test_inference.py
|   +-- test_api.py
|
+-- scripts/
|
+-- notebooks/
|
+-- artifacts/
|
+-- Dockerfile
+-- docker-compose.yml
+-- dvc.yaml
+-- params.yaml
+-- pyproject.toml
+-- .gitignore
+-- .dockerignore
+-- README.md
```

The exact structure may evolve during implementation. Changes should be documented in this README rather than made silently.

---

# 7. Development Phases

## Phase 0 — Project Foundation

### Objectives

Create a clean and reproducible development environment.

### Tasks

- [ ] Create Git repository
- [ ] Create Conda environment
- [ ] Install Python dependencies
- [ ] Install PyTorch
- [ ] Verify PyTorch installation
- [ ] Check CPU availability
- [ ] Check CUDA/GPU availability
- [ ] Create repository structure
- [ ] Create `.gitignore`
- [ ] Create `pyproject.toml`
- [ ] Establish initial README
- [ ] Make initial Git commit

### Completion Criteria

We can successfully run a small PyTorch script and confirm:

```text
Python version
PyTorch version
CUDA availability
GPU name if available
```

---

# 8. Phase 1 — Dataset

## Objective

Obtain a realistic customer-support classification dataset and establish a reproducible data pipeline.

### Tasks

- [ ] Select dataset
- [ ] Document dataset source/license
- [ ] Download raw data
- [ ] Store raw data appropriately
- [ ] Inspect schema
- [ ] Inspect class distribution
- [ ] Check missing values
- [ ] Check duplicates
- [ ] Check malformed records
- [ ] Validate labels
- [ ] Create train/validation/test split
- [ ] Version dataset with DVC

### Our primary dataset

Primary dataset: BANKING77
Source: PolyAI / Hugging Face
License: CC BY 4.0
13,083 examples / 77 intents
Official split: 10,003 train / 3,080 test
Task: fine-grained single-domain customer-service intent classification.

### Secondary/Future Dataset

Bitext customer support- may be used to test whether the pipeline generalizes to a different customer-support dataset with more examples and a different data=generation process

### Requirements

The split must avoid data leakage.

Target:

```text
train
validation
test
```

with the split strategy documented in the project.

---

# 9. Phase 2 — Data Validation and Preprocessing

## Objective

Create deterministic preprocessing.

### Tasks

- [ ] Text cleaning where justified
- [ ] Label normalization
- [ ] Label encoding
- [ ] Remove/handle invalid records
- [ ] Tokenization
- [ ] Padding/truncation
- [ ] Maximum sequence length configuration
- [ ] Class distribution analysis
- [ ] Reproducible processing

Important principle:

> Do not aggressively clean text merely because it looks messy. Preserve information that may be useful to the Transformer.

---

# 10. Phase 3 — Hugging Face + PyTorch Model

## Objective

Fine-tune a pretrained Transformer.

Initial model:

```text
distilbert-base-uncased
```

Core components:

```python
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes
)
```

### PyTorch concepts to learn

- `torch.Tensor`
- device management
- `nn.Module`
- forward pass
- loss
- gradients
- optimizer
- scheduler
- Dataset
- DataLoader
- `model.train()`
- `model.eval()`
- `torch.no_grad()`
- checkpoints
- GPU training
- mixed precision

---

# 11. Training Strategy

We should understand both approaches.

## Approach A — Native PyTorch Training Loop

We should first understand the underlying mechanics:

```text
Dataset
  |
DataLoader
  |
Batch
  |
Model
  |
Prediction
  |
Loss
  |
Backward
  |
Optimizer
  |
Scheduler
```

This is important for PyTorch understanding.

## Approach B — Hugging Face Trainer

After the baseline training loop works, evaluate the Hugging Face `Trainer` abstraction.

The project should document:

- when the abstraction is useful
- what it hides
- what customization it allows
- when a custom training loop is preferable

We should not blindly use a framework abstraction without understanding what it does.

---

# 12. Training Features

The training pipeline should eventually support:

- [ ] Configurable learning rate
- [ ] Configurable batch size
- [ ] Configurable epochs
- [ ] Weight decay
- [ ] Optimizer selection
- [ ] Learning-rate scheduler
- [ ] Random seed
- [ ] Maximum sequence length
- [ ] Gradient accumulation
- [ ] Gradient clipping
- [ ] Mixed precision
- [ ] Early stopping
- [ ] Checkpointing
- [ ] Resume-from-checkpoint
- [ ] GPU support
- [ ] CPU fallback

---

# 13. Reproducibility

Every model should be traceable to:

```text
Git commit
+
DVC dataset version
+
Hugging Face model ID
+
Hugging Face model revision
+
Tokenizer
+
Configuration
+
Random seed
+
Python version
+
PyTorch version
+
Transformers version
+
Training run
+
MLflow experiment
```

Record environment information during training.

A seed should be configurable rather than hard-coded throughout the codebase.

Important:

> Setting random seeds improves reproducibility but does not guarantee identical results across every hardware/software configuration.

---

# 14. Checkpointing

Checkpoints should support recovery from interrupted training.

Track at minimum:

```text
epoch/step
model state
optimizer state
scheduler state
training loss
validation metrics
configuration
```

Desired capability:

```text
Training
   |
   +---- interruption
             |
             v
        Load checkpoint
             |
             v
       Resume training
```

---

# 15. Experiment Tracking with MLflow

Every training run should record:

## Parameters

```text
model name
model revision
learning rate
batch size
epochs
weight decay
optimizer
scheduler
max sequence length
seed
gradient accumulation
mixed precision
```

## Metrics

```text
training loss
validation loss
accuracy
precision
recall
macro F1
weighted F1
per-class F1
```

## Artifacts

```text
model
tokenizer
configuration
confusion matrix
evaluation report
training plots
checkpoint information
```

---

# 16. Model Registry

Lifecycle:

```text
Training
   |
   v
Candidate Model
   |
   v
Evaluation
   |
   v
Quality Gate
   |
   +---- FAIL ----> Reject
   |
   +---- PASS ----> Registry
                         |
                         v
                      Staging
                         |
                         v
                     Production
```

The registered model must retain enough metadata to determine exactly how it was produced.

---

# 17. Model Quality Gates

We will not deploy a model solely because training completed successfully.

Example initial configuration:

```yaml
quality_gates:
  macro_f1: 0.85
  weighted_f1: 0.90
  minimum_class_recall: 0.70
  max_inference_latency_ms: 300
```

These are initial engineering thresholds, not final values.

They must be adjusted based on:

- dataset characteristics
- class imbalance
- business requirements
- baseline performance
- inference environment

A model must pass the relevant evaluation gates before production promotion.

---

# 18. Evaluation

Evaluation should include more than accuracy.

Required:

- [ ] Accuracy
- [ ] Precision
- [ ] Recall
- [ ] Macro F1
- [ ] Weighted F1
- [ ] Per-class metrics
- [ ] Confusion matrix
- [ ] Classification report
- [ ] Inference latency
- [ ] Model size
- [ ] Number of parameters

For imbalanced datasets, macro F1 and class-level metrics should receive particular attention.

---

# 19. Hugging Face Model Management

The project should record:

```text
model ID
model revision
tokenizer ID
tokenizer revision
configuration
number of labels
label mapping
```

For example:

```text
id2label
label2id
```

must be stored with the model configuration.

The production system should not rely on an undocumented label ordering.

---

# 20. PEFT / LoRA

After the full fine-tuning baseline works, implement parameter-efficient fine-tuning.

Target technologies:

- PEFT
- LoRA
- QLoRA where appropriate

Comparison:

```text
Full Fine-tuning
       |
       +-- all/most model parameters updated

LoRA
       |
       +-- base model frozen
       |
       +-- small adapter parameters trained
```

Compare:

```text
training time
GPU memory
model artifact size
performance
inference behavior
deployment complexity
```

The goal is to understand the engineering tradeoffs, not merely add LoRA because it is popular.

---

# 21. Inference Pipeline

Training and inference must be separate concerns.

Target flow:

```text
Request
  |
  v
Input validation
  |
  v
Text preprocessing
  |
  v
Tokenizer
  |
  v
Model
  |
  v
Softmax / prediction
  |
  v
Response
```

Inference must use:

```python
model.eval()
```

and appropriate inference/no-gradient behavior.

---

# 22. FastAPI

Endpoints:

```text
GET  /health
GET  /metadata
POST /predict
```

Example request:

```json
{
  "text": "I was charged twice for the same transaction."
}
```

Example response:

```json
{
  "prediction": "billing_issue",
  "confidence": 0.9472,
  "model_version": "3",
  "model_id": "customer-ticket-classifier"
}
```

API requirements:

- [ ] Pydantic validation
- [ ] Error handling
- [ ] Structured logging
- [ ] Request ID
- [ ] Model version reporting
- [ ] Health check
- [ ] Latency measurement
- [ ] Safe model loading
- [ ] Configurable model location

---

# 23. Docker

The inference application must be containerized.

Requirements:

- [ ] Production Dockerfile
- [ ] `.dockerignore`
- [ ] Environment configuration
- [ ] Non-root execution where practical
- [ ] Health check
- [ ] Reproducible dependency installation
- [ ] Image versioning
- [ ] Local container test

Potential deployment model:

```text
Docker image
     |
     v
AWS ECR
     |
     v
AWS EC2
```

GPU deployment can be added later if the chosen model/infrastructure requires it.

---

# 24. CI/CD

GitHub Actions should eventually automate:

```text
Git push
   |
   v
Install dependencies
   |
   v
Lint
   |
   v
Unit tests
   |
   v
Integration tests
   |
   v
Build Docker image
   |
   v
Security/quality checks
   |
   v
Push image
   |
   v
Deploy
```

Training should be separated from ordinary application CI where appropriate.

A code change should not automatically trigger an expensive GPU training job unless explicitly configured to do so.

---

# 25. Testing Strategy

## Unit Tests

Test:

- preprocessing
- label mapping
- tokenization
- model initialization
- tensor shapes
- inference
- configuration

Example:

```python
x = torch.randn(4, 10)
output = model(x)

assert output.shape == (4, 1)
```

The actual assertion will match the selected model/task.

## Integration Tests

Test:

```text
API
  |
model
  |
tokenizer
  |
prediction
```

## Data Tests

Validate:

- schema
- required columns
- label values
- missing values
- duplicates
- unexpected data types

---

# 26. Monitoring

## Infrastructure

Monitor:

```text
CPU
RAM
GPU utilization
GPU memory
disk
network
```

## API

Monitor:

```text
request count
latency
p50
p95
p99
error rate
5xx rate
```

## Model

Monitor:

```text
prediction distribution
confidence distribution
class distribution
model performance where labels become available
```

## Data

Monitor:

```text
input length
token length
missing values
class distribution
input drift
```

---

# 27. Evidently

Evidently will be used for model/data monitoring where appropriate.

Potential monitoring flow:

```text
Production Requests
        |
        v
Logged Predictions
        |
        v
Reference vs Current Data
        |
        v
Evidently
        |
        v
Drift Report
        |
        v
Alert / Retraining Decision
```

Important:

> Drift detection does not automatically mean the model is performing badly. It is a signal that should be interpreted alongside model/business performance.

---

# 28. Retraining Strategy

Potential future workflow:

```text
Production
    |
    v
Monitoring
    |
    v
Drift / performance degradation
    |
    v
Trigger retraining
    |
    v
New training run
    |
    v
MLflow
    |
    v
Evaluation
    |
    v
Quality gate
    |
    +---- FAIL ----> Keep production model
    |
    +---- PASS ----> Candidate model
                         |
                         v
                       Deploy
```

Automated retraining should only be introduced after the manual pipeline is reliable.

---

# 29. AWS Architecture

Initial target:

```text
                    AWS
                     |
        +------------+-------------+
        |            |             |
        v            v             v
       S3           ECR           EC2
        |            |             |
    datasets     Docker image   FastAPI
    artifacts                    inference
        |
        v
     MLflow
```

Potential services:

- S3
- ECR
- EC2
- IAM
- CloudWatch

The exact architecture will depend on model size, inference requirements, budget, and whether GPU infrastructure is justified.

---

# 30. Security Considerations

The project should demonstrate basic production security practices.

- [ ] No secrets committed to Git
- [ ] Environment variables for credentials
- [ ] AWS IAM least privilege where practical
- [ ] API input validation
- [ ] Dependency pinning/management
- [ ] Container security
- [ ] Avoid logging sensitive customer text unnecessarily
- [ ] Consider PII handling
- [ ] Avoid storing raw production requests without a defined purpose

If the selected dataset contains sensitive information, anonymization and data handling requirements must be documented.

---

# 31. Configuration

Avoid hard-coding important parameters throughout the source code.

Example:

```yaml
model:
  name: distilbert-base-uncased
  max_length: 256

training:
  learning_rate: 0.00002
  batch_size: 16
  epochs: 3
  weight_decay: 0.01
  seed: 42

evaluation:
  macro_f1_threshold: 0.85
  weighted_f1_threshold: 0.90
```

Configuration should be loaded centrally and passed to the relevant components.

---

# 32. Experimentation Strategy

The project should establish a baseline before optimization.

Example progression:

```text
Experiment 1
DistilBERT + full fine-tuning
        |
        v
Baseline

Experiment 2
Hyperparameter tuning
        |
        v
Improved baseline

Experiment 3
Alternative Transformer
        |
        v
Comparison

Experiment 4
LoRA
        |
        v
Parameter-efficient model

Experiment 5
Production optimization
        |
        v
Latency / memory / cost analysis
```

Do not optimize blindly.

Every experiment should have:

```text
hypothesis
configuration
dataset version
model version
metrics
conclusion
```

---

# 33. Model Comparison

Models should be compared using documented metrics rather than intuition.

Record:

```text
Model
Dataset version
Training configuration
Macro F1
Weighted F1
Minimum class recall
Parameter count
Model size
Training time
Peak memory
Inference latency
Throughput
```

Do not select a model solely on one metric. The final production choice should be based on the documented requirements of the application.

---

# 34. Production Performance

Eventually measure:

```text
                    MODEL
                      |
        +-------------+-------------+
        |             |             |
        v             v             v
     Accuracy      Latency        Memory
        |             |             |
        v             v             v
     Quality        p95/p99      RAM/VRAM
```

We should investigate:

- batch inference
- dynamic padding
- sequence length
- CPU vs GPU inference
- model quantization
- ONNX/export options where useful
- Torch compilation/optimization where appropriate
- caching
- concurrency

Optimization must be measured rather than assumed.

---

# 35. Model Optimization Roadmap

Potential future path:

```text
PyTorch
   |
   +--> Hugging Face Transformers
   |
   +--> torch.compile
   |
   +--> ONNX
   |
   +--> TensorRT
   |
   +--> Quantization
```

These are optional optimization stages.

They should only be introduced after establishing a working baseline and measuring the bottleneck.

---

# 36. Observability

Every production prediction should be traceable without unnecessarily storing sensitive text.

Potential event:

```json
{
  "request_id": "abc-123",
  "timestamp": "2026-09-29T16:00:00Z",
  "model_id": "customer-ticket-classifier",
  "model_version": "3",
  "prediction": "billing_issue",
  "confidence": 0.9472,
  "latency_ms": 84
}
```

Do not log raw customer content by default.

---

# 37. Failure Scenarios to Test

The system should eventually be tested against:

- Invalid API input
- Empty text
- Extremely long text
- Missing model artifact
- Corrupt model artifact
- Tokenizer/model mismatch
- Unknown label mapping
- Out-of-memory conditions
- CPU fallback
- GPU unavailable
- MLflow unavailable
- S3 unavailable
- Invalid configuration
- Bad dataset schema
- Missing labels
- Model quality below threshold
- Docker container startup failure

The purpose is to demonstrate production thinking, not merely successful execution.

---

# 38. Definition of Done

The project is not complete when the model reaches a good F1 score.

The project is complete when we can demonstrate:

```text
[ ] Versioned dataset
[ ] Reproducible preprocessing
[ ] Hugging Face pretrained model
[ ] PyTorch fine-tuning
[ ] Validation/evaluation
[ ] Checkpointing
[ ] Experiment tracking
[ ] MLflow model registry
[ ] Quality gates
[ ] FastAPI inference
[ ] Docker container
[ ] Automated tests
[ ] GitHub Actions CI/CD
[ ] AWS deployment
[ ] Monitoring
[ ] Drift detection
[ ] Model versioning
[ ] Reproducible model lineage
[ ] Documentation
```

---

# 39. Final Portfolio Story

The project should eventually be explainable in an interview as:

> "I built a production-oriented NLP classification system using PyTorch and Hugging Face Transformers. I versioned the dataset with DVC, fine-tuned and evaluated pretrained Transformer models, tracked experiments and artifacts with MLflow, implemented model quality gates and registry-based promotion, exposed inference through FastAPI, containerized the service with Docker, deployed it to AWS, and added monitoring and drift detection. The pipeline was designed so model, data, code, configuration, and environment could be traced for reproducibility."

That is the story this repository should support.

---

# 40. Learning Objectives

By the end of this project, I should be comfortable with:

## PyTorch

- [ ] Tensors
- [ ] Devices
- [ ] Autograd
- [ ] `nn.Module`
- [ ] Dataset
- [ ] DataLoader
- [ ] Training loops
- [ ] Evaluation loops
- [ ] Optimizers
- [ ] Schedulers
- [ ] Checkpoints
- [ ] Mixed precision
- [ ] GPU training

## Hugging Face

- [ ] Tokenizers
- [ ] Auto classes
- [ ] Pretrained models
- [ ] Sequence classification
- [ ] Model configuration
- [ ] Trainer
- [ ] Hugging Face Hub
- [ ] Model revisions
- [ ] PEFT
- [ ] LoRA

## MLOps

- [ ] Git
- [ ] DVC
- [ ] MLflow
- [ ] Model Registry
- [ ] FastAPI
- [ ] Docker
- [ ] GitHub Actions
- [ ] AWS
- [ ] Monitoring
- [ ] Evidently
- [ ] Model/data drift
- [ ] CI/CD

---

# 41. Current Status

## Phase

**Phase 0 — Project Foundation**

## Current task

```text
Create the repository
        |
        v
Create Conda environment
        |
        v
Install PyTorch + Hugging Face dependencies
        |
        v
Verify environment
        |
        v
Initialize project structure
```

## Progress

- [x] Project concept defined
- [x] MLOps architecture defined
- [x] Technology stack defined
- [x] Repository structure defined
- [x] Development roadmap defined
- [x] Production requirements defined
- [ ] Repository created
- [ ] Conda environment created
- [ ] Dependencies installed
- [ ] PyTorch verified
- [ ] GPU/CUDA verified
- [ ] Initial Git commit

---

# 42. Project Rules

These rules keep the project from becoming a collection of disconnected tutorials.

### Rule 1 — Production first

Every component should eventually have a reason to exist in a production workflow.

### Rule 2 — Understand abstractions

Before using a high-level library abstraction, understand what it is doing underneath.

### Rule 3 — Reproducibility

A model must be traceable to its code, data, configuration, environment, and training run.

### Rule 4 — Measure before optimizing

Do not add optimization technologies simply because they are popular.

### Rule 5 — No hard-coded secrets

Credentials and API keys must never enter Git.

### Rule 6 — No silent architecture changes

If repository structure or major technology decisions change, update this README.

### Rule 7 — Every experiment is documented

Record the hypothesis, configuration, dataset version, metrics, and conclusion.

### Rule 8 — Don't over-engineer early

Build the simplest correct baseline first, then introduce production complexity deliberately.

### Rule 9 — Tests are part of the system

Tests should be developed alongside the application rather than added at the very end.

### Rule 10 — The README is the project memory

When development pauses and resumes later, this file should tell us:

- what we built
- why we built it
- what changed
- what works
- what remains
- what the next task is

---

# 43. Change Log

## 2026-09-29

### Initial project specification

Established:

- PyTorch + Hugging Face architecture
- Customer Support Ticket Classification use case
- DVC data versioning
- MLflow experiment tracking and model registry
- FastAPI inference
- Docker packaging
- AWS deployment target
- GitHub Actions CI/CD
- Evidently monitoring
- Drift detection
- PEFT/LoRA roadmap
- Reproducibility requirements
- Quality gates
- Testing strategy
- Project repository structure

Next step:

**Phase 0 — create the environment and verify PyTorch/Hugging Face installation.**

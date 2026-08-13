# DLGenAI Project — T3 2025

**Multi-Label Emotion Classification: Scratch Neural Networks vs Transformer Models**

Course Code: DLGenAI — IIT Madras BS Degree in Data Science and Applications (September 2025 term)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License: Academic](https://img.shields.io/badge/license-academic-lightgrey.svg)](#license)

---

## Overview

This project implements a modular, end-to-end deep learning pipeline for detecting five emotions in short text spans:

`anger` · `fear` · `joy` · `sadness` · `surprise`

It's a **multi-label** problem — a single sample can carry more than one emotion at once — so every model outputs independent sigmoid probabilities per label rather than a single softmax class.

The pipeline compares two very different approaches side by side:

| Approach | What it tests |
|---|---|
| Scratch BiLSTM | How far you get with embeddings + recurrence, no pretraining |
| Fine-tuned BERT / RoBERTa | How much pretraining buys you on the same data/split |

Everything — data loading, training, inference, reporting, and optional model upload — runs from a single unified entry point, and the repo is structured to work identically on a local machine or inside a Kaggle notebook.

---

## Author

**Gaurav Tomar**
Student ID: `23f1000805`
Program: BS Degree in Data Science and Applications — IIT Madras

## Live Deployment

🚀 Try the deployed app here: [huggingface.co/spaces/23f1000805/DLGenAI-project-deployment](https://huggingface.co/spaces/23f1000805/DLGenAI-project-deployment)

---

## Project Structure

```
emotion-classification-project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/                          # Not tracked in git
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── scripts/
│   ├── config.py                  # Paths, constants, device, W&B settings
│   ├── data_loader.py             # CSV loading + stratified-ish train/val split
│   ├── vocab_scratch.py           # Vocabulary building + text encoding for BiLSTM
│   ├── scratch_model.py           # Embedding + BiLSTM + linear head
│   ├── scratch_train.py           # Training loop for the scratch model
│   ├── transformer_train.py       # BERT / RoBERTa fine-tuning via HF Trainer
│   ├── inference_utils.py         # Shared inference / thresholding helpers
│   ├── uploader.py                # Optional: push models to KaggleHub
│   ├── reporting.py                # Metrics summary, plots, text observations
│   ├── train.py                   # Entry point: trains all three models
│   └── inference.py               # Entry point: runs inference for all models
│
├── notebooks/
│   ├── dl-23f1000805-notebook-t32025-train.ipynb   # preprocessing + training
│   └── dl-23f1000805-notebook-t32025-infer.ipynb   # inference + submission
│
└── project_outputs/                # Created at runtime — models, logs, submissions
```

---

## Features

### 1. Scratch BiLSTM Model
- Custom vocabulary builder (frequency-based, with `<unk>`/`<pad>` handling)
- Embedding → BiLSTM → linear classification head
- Trained entirely from scratch, no pretrained weights — a clean baseline for "what does the architecture alone buy you"

### 2. Transformer-Based Models
- `bert-base-uncased`
- `roberta-base`
- Fine-tuned via the Hugging Face `Trainer` API
- Multi-label output via `BCEWithLogitsLoss` on sigmoid activations (not softmax/cross-entropy)

### 3. Unified Training (`train.py`)
Runs, in sequence:
1. Scratch BiLSTM training
2. BERT fine-tuning
3. RoBERTa fine-tuning

Saves checkpoints, logs, and submission files to `project_outputs/`.

### 4. Unified Inference (`inference.py`)
Loads all three trained models and produces:
- `submission_scratch.csv`
- `submission_bert.csv`
- `submission_roberta.csv`

### 5. Reporting (`reporting.py`)
- CSV summary of per-model scores (precision / recall / F1-micro / F1-macro)
- Bar chart comparing F1 across models
- Line plot for per-label breakdown
- Plain-text observation summary

### 6. Optional KaggleHub Upload (`uploader.py`)
- Upload full Hugging Face model folders
- Upload raw `.pt` state-dict files
- Automatic dataset/model versioning through KaggleHub

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd emotion-classification-project
pip install -r requirements.txt
```

Core dependencies:

```
torch>=2.0
transformers>=4.40
scikit-learn
pandas
numpy
matplotlib
kagglehub      # optional, only if using uploader.py
wandb          # optional, only if W&B logging is enabled
```

### 2. Environment variables (optional but recommended)

If you hit Hugging Face Hub slowness during model download/upload, disable Xet storage:

```bash
export HF_HUB_DISABLE_XET=1
```

If using Weights & Biases, set your key before training:

```bash
export WANDB_API_KEY=your_key_here
```

---

## How to Run — Local

### Step 1 — Place the dataset

```
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### Step 2 — Train

```bash
python scripts/train.py
```

This loads the dataset, trains all three models in sequence, and writes checkpoints + submissions to `project_outputs/`.

### Step 3 — Run inference

```bash
python scripts/inference.py
```

Produces `submission_scratch.csv`, `submission_bert.csv`, and `submission_roberta.csv` inside `project_outputs/`.

---

## How to Run — Kaggle

### Step 1 — Upload the project
Upload a ZIP of the repo, or connect the GitHub repo directly as a Kaggle Dataset/Notebook source.

### Step 2 — Extract

```bash
!unzip project.zip -d project
```

### Step 3 — Copy the competition data in

```bash
!cp /kaggle/input/2025-sep-dl-gen-ai-project/* project/data/
```

### Step 4 — Train

```bash
!python project/scripts/train.py
```

### Step 5 — Infer

```bash
!python project/scripts/inference.py
```

> Compatible with the standard Kaggle Python image — no extra system packages required for the transformer models. GPU accelerator is strongly recommended for BERT/RoBERTa fine-tuning.

---

## Expected Model Performance

| Model | Type | Expected F1-Micro | Notes |
|---|---|---|---|
| Scratch BiLSTM | From-scratch | ~0.30 – 0.60 | Highly sensitive to vocab size, embedding dim, seed |
| BERT-base | Transformer | ~0.84 – 0.86 | 2–3 epochs usually enough before overfitting |
| RoBERTa-base | Transformer | ~0.85 – 0.87 | Marginal gain over BERT, more sensitive to LR |

Actual numbers depend on hyperparameters, random seed, and the exact train/val split — treat these as ballpark references, not guarantees.

---

## Reporting Example

```python
from scripts.reporting import save_summary, plot_bar, plot_line, print_observations

df, summary_path = save_summary(results, "project_outputs")
plot_bar(df)
plot_line(df)
print_observations(df)
```

## KaggleHub Upload Example

```python
from scripts.uploader import upload_folder_to_kagglehub

upload_folder_to_kagglehub(
    local_dir="project_outputs/bert_base_hf",
    model_slug="username/emotion-models",
    framework="pytorch",
    notes="BERT-base fine-tuned on DLGenAI emotion dataset",
)
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Hugging Face download/upload hangs | `export HF_HUB_DISABLE_XET=1` |
| BatchNorm/Dropout crash on last batch | Set `drop_last=True` in the DataLoader |
| Kaggle secret not found | Double-check exact key casing in Kaggle Secrets |
| CUDA OOM on transformer training | Lower `per_device_train_batch_size`, enable gradient accumulation |
| Submission columns mismatched | Confirm label order matches `sample_submission.csv` exactly |

---

## Learning Outcomes

- Built a complete NLP pipeline from raw text to submission-ready CSVs
- Implemented and reasoned about multi-label classification (sigmoid + BCE, not softmax)
- Directly compared a from-scratch architecture against pretrained transformers
- Used the Hugging Face `Trainer` API for fine-tuning
- Structured a modular ML project: config, data, training, inference, reporting, and upload as separate concerns
- Wrote code that runs unmodified on both local machines and Kaggle notebooks

---

## Technologies Used

Python · PyTorch · Hugging Face Transformers · scikit-learn · pandas · NumPy · Matplotlib · KaggleHub (optional) · Weights & Biases (optional)

---

## License

This project is for educational and academic purposes under the DLGenAI course (September 2025 term), IIT Madras BS Degree in Data Science and Applications.

---

## Acknowledgements

- IIT Madras BS Data Science program
- Kaggle datasets and runtime environment
- Hugging Face Transformers library
- PyTorch team

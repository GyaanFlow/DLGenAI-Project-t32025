# DLGenAI Project — T3 2025

**Multi-Label Emotion Classification: Scratch BiLSTM vs Fine-Tuned Transformers**

Course Code: DLGenAI — IIT Madras BS Degree in Data Science and Applications (September 2025 term)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Runtime-Kaggle-20BEFF.svg)](https://www.kaggle.com/)
[![License: Academic](https://img.shields.io/badge/license-academic-lightgrey.svg)](#license)

🚀 **Live app:** [huggingface.co/spaces/23f1000805/DLGenAI-project-deployment](https://huggingface.co/spaces/23f1000805/DLGenAI-project-deployment)

---

## Overview

This project detects five emotions in short text spans:

`anger` · `fear` · `joy` · `sadness` · `surprise`

It's a **multi-label** problem — a sample can carry more than one emotion at once — so every model outputs independent sigmoid probabilities per label (`BCEWithLogitsLoss`), not a single softmax class.

Three models are trained and compared on the same train/validation split:

| Model | Type | What it tests |
|---|---|---|
| Scratch BiLSTM | Embedding + BiLSTM, trained from zero | How far a simple recurrent architecture gets with no pretraining |
| `bert-base-uncased` | Fine-tuned transformer | How much pretraining buys you on the same data |
| `roberta-base` | Fine-tuned transformer | Whether RoBERTa's pretraining recipe adds further gains |

The pipeline is **Kaggle-native**: it reads competition data from `/kaggle/input/...`, pulls secrets via `kaggle_secrets.UserSecretsClient`, and writes all artifacts to `/kaggle/working/project_outputs`. Trained models are optionally versioned and pushed to **KaggleHub**, then re-downloaded for inference and final deployment to Hugging Face Spaces.

---

## Author

**Gaurav Tomar**
Student ID: `23f1000805`
Program: BS Degree in Data Science and Applications — IIT Madras

---

## Repository Structure

```
DLGenAI-Project-t32025/
│
├── README.md
│
├── data/
│   ├── 2025-sep-dl-gen-ai-project.zip   # raw competition archive
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── notebooks/
│   ├── dl-23f1000805-notebook-t32025 (1).ipynb
│   └── dl-23f1000805-notebook-t32025 (2).ipynb
│
├── scripts/                              # imported inside Kaggle as `src.*`
│   ├── config.py                         # Kaggle secrets, device, W&B setup, label list
│   ├── data_loader.py                    # CSV loading + train/val split
│   ├── vocab_scratch.py                  # vocab builder + tokenizer + Dataset for BiLSTM
│   ├── scratch_model.py                  # SimpleBiLSTM architecture
│   ├── scratch_train.py                  # training loop for the scratch model
│   ├── transformer_train.py              # BERT / RoBERTa fine-tuning via HF Trainer
│   ├── train.py                          # entry point: trains all 3 models in sequence
│   ├── inference.py                      # loads trained models, runs predictions
│   ├── reporting.py                      # metrics summary + plots + observations
│   └── uploader.py                       # push trained models to KaggleHub
│
├── project_outputs/
│   ├── submission_scratch (4).csv
│   ├── submission_bert_base_uncased (1).csv
│   └── submission_roberta_base (1).csv
│
└── report/
    └── 23f1000805_DG_T32025 (1).pdf      # final written project report
```

> **Note on imports:** the scripts use `from src.config import ...` etc. Inside Kaggle, the `scripts/` folder is copied/mounted as `src/` in the working directory (or added to `sys.path` under that name) before `train.py` / `inference.py` are run — that's why the folder is named `scripts/` on disk but imported as `src` in code.

---

## Pipeline Details

### 1. Config (`config.py`)
- Reads `KAGGLE_USERNAME` and `WANDB_API_KEY` from Kaggle Secrets (`UserSecretsClient`)
- Picks `cuda` if available, else `cpu`
- Defines the label order: `["anger", "fear", "joy", "sadness", "surprise"]`
- Logs into Weights & Biases if a key is present, otherwise falls back to **offline W&B mode** so training never hard-fails on a missing key
- All outputs go to `/kaggle/working/project_outputs`

### 2. Scratch BiLSTM (`scratch_model.py`, `vocab_scratch.py`, `scratch_train.py`)
- Vocabulary: whitespace tokenization, top **10,000** most frequent words, `<pad>`/`<unk>` reserved
- Sequences padded/truncated to **max_len = 50**
- Architecture: `Embedding(128) → BiLSTM(hidden=128) → mean-pool over time → Linear(256 → 5)`
- Training: **100 epochs**, batch size 64, `AdamW` at `lr=3e-4`, `BCEWithLogitsLoss`
- No early stopping — instead, the checkpoint with the best validation **F1-micro** is saved each epoch (`best_scratch_state.pt` + full model `best_scratch_full.pt`)

### 3. Transformer fine-tuning (`transformer_train.py`)
- Tokenization: `padding="max_length"`, `truncation=True`, `max_length=128`
- `AutoModelForSequenceClassification` with `problem_type="multi_label_classification"` (sigmoid + BCE under the hood)
- Hugging Face `Trainer` with:
  - `num_train_epochs=8`, `learning_rate=2e-5`
  - `per_device_train_batch_size=16`, `per_device_eval_batch_size=32`
  - `warmup_ratio=0.1`, `weight_decay=0.02`
  - `eval_strategy="epoch"`, `save_strategy="epoch"`, `load_best_model_at_end=True` on **F1-micro**
  - `fp16=True` automatically when a GPU is available
  - W&B reporting enabled automatically if `USE_WANDB` is set
- After training: predicts on the held-out validation set, writes `submission_<tag>.csv`, and saves the full HF model + tokenizer to `project_outputs/<tag>_hf/`

### 4. Unified training (`train.py`)
Runs, in order: scratch BiLSTM → `bert-base-uncased` → `roberta-base`, all against the same train/val split (80/20, `random_state=42`).

### 5. Inference (`inference.py`)
Loads all three trained models **from their KaggleHub model-version folders** (hardcoded per-run paths, e.g. `/kaggle/input/emotion-models-<timestamp>/pytorch/.../1`) and reproduces predictions for each:
- Scratch model: rebuilds the vocab from `train.csv`, loads `best_scratch_state.pt`, runs a forward pass, thresholds at `0.5`
- BERT / RoBERTa: loads each fine-tuned `AutoModelForSequenceClassification` + tokenizer, runs a forward pass, thresholds at `0.5`

> Update `SCRATCH_FOLDER`, `BERT_FOLDER`, and `ROBERTA_FOLDER` at the top of `inference.py` to point at whichever KaggleHub model version you want to score against — these are not resolved automatically.

### 6. Reporting (`reporting.py`)
- `save_summary(results, out_dir)` → writes `model_summary.csv` from a list of `{model, f1_micro, submission}` dicts
- `plot_bar(df)` / `plot_line(df)` → F1-micro comparison charts
- `print_observations(df)` → plain-text takeaways (transformers outperform the scratch model, scratch model helps build pipeline intuition, best model should be used for deployment)

### 7. KaggleHub upload (`uploader.py`)
- `upload_folder_kagglehub(local_dir, handle, notes="")` → uploads an entire folder (e.g. a saved HF model) as a new KaggleHub model version, timestamp-suffixed
- `upload_pt_file(pt_path, handle, notes="")` → uploads a single `.pt` file (e.g. the scratch model's state dict) the same way
- Both silently no-op with a message if `kagglehub` isn't installed (i.e. outside Kaggle)

---

## Running on Kaggle

1. Add the competition dataset (`2025-sep-dl-gen-ai-project`) as a data source on the notebook.
2. Copy `scripts/` into the working directory as `src/` (or add `scripts/` to `sys.path` under that alias) so the `from src.* import ...` statements resolve.
3. Set `KAGGLE_USERNAME` and (optionally) `WANDB_API_KEY` under **Add-ons → Secrets**.
4. Run training:
   ```python
   %run src/train.py
   ```
5. Upload the resulting model folders/state dicts to KaggleHub if you want them accessible for a separate inference run:
   ```python
   from src.uploader import upload_folder_kagglehub, upload_pt_file

   upload_folder_kagglehub("project_outputs/bert_base_hf", "your-username/emotion-models", notes="BERT fine-tuned")
   upload_pt_file("project_outputs/scratch_model/best_scratch_state.pt", "your-username/emotion-models", notes="Scratch BiLSTM")
   ```
6. In a fresh notebook (or the same one), attach the KaggleHub model version as an input, update the three `*_FOLDER` paths in `inference.py`, and run:
   ```python
   %run src/inference.py
   ```
7. Summarize results:
   ```python
   from src.reporting import save_summary, plot_bar, plot_line, print_observations

   df, path = save_summary(results, "project_outputs")
   plot_bar(df)
   plot_line(df)
   print_observations(df)
   ```

### Adapting to run locally
The scripts assume Kaggle (`kaggle_secrets`, `/kaggle/input`, `/kaggle/working`). To run outside Kaggle you'd need to:
- Replace `UserSecretsClient` calls in `config.py` with `os.environ.get(...)` or a `.env` file
- Point `DATA_DIR` in `train.py` / `inference.py` at a local `data/` folder instead of `/kaggle/input/...`
- Change `OUT_DIR` to a local path instead of `/kaggle/working/project_outputs`
- Skip `uploader.py` (KaggleHub-only) or swap it for a Hugging Face Hub upload

---

## Live Deployment

The best-performing model is deployed as a Gradio app on Hugging Face Spaces:

🚀 [huggingface.co/spaces/23f1000805/DLGenAI-project-deployment](https://huggingface.co/spaces/23f1000805/DLGenAI-project-deployment)

---

## Results

Submissions for all three models are checked into `project_outputs/`:
- `submission_scratch (4).csv`
- `submission_bert_base_uncased (1).csv`
- `submission_roberta_base (1).csv`

Full write-up, methodology, and analysis are in the project report:
📄 `report/23f1000805_DG_T32025 (1).pdf`

General pattern observed (see `reporting.print_observations`): transformer models (BERT/RoBERTa) clearly outperform the from-scratch BiLSTM, the scratch model is mainly useful for building pipeline/architecture intuition, and BERT/RoBERTa reach strong scores within just a few epochs of fine-tuning.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Copy/alias `scripts/` as `src/` in the working directory, or add it to `sys.path` before importing |
| `kaggle_secrets` import fails locally | This module only exists inside Kaggle notebooks — see "Adapting to run locally" above |
| Hugging Face download/upload hangs | `export HF_HUB_DISABLE_XET=1` |
| BatchNorm/last-batch crash in `DataLoader` | Set `drop_last=True` |
| Kaggle secret not found | Double-check exact key casing (`KAGGLE_USERNAME`, `WANDB_API_KEY`) in Kaggle Secrets |
| CUDA OOM during transformer fine-tuning | Lower `per_device_train_batch_size` in `transformer_train.py`, or add gradient accumulation |
| `inference.py` can't find model files | Update `SCRATCH_FOLDER` / `BERT_FOLDER` / `ROBERTA_FOLDER` to the correct KaggleHub model-version path for your run |

---

## Learning Outcomes

- Built a complete multi-label NLP pipeline from raw competition data to submission-ready CSVs
- Implemented and reasoned about multi-label classification (sigmoid + BCE, not softmax/cross-entropy)
- Directly compared a from-scratch recurrent architecture against pretrained transformers on identical splits
- Used the Hugging Face `Trainer` API end-to-end: tokenization, custom `compute_metrics`, best-checkpoint selection
- Built a Kaggle-native MLOps loop: train → checkpoint → push to KaggleHub → pull for inference → deploy to Hugging Face Spaces
- Structured a modular project: config, data, training, inference, reporting, and upload as separate concerns

---

## Technologies Used

Python · PyTorch · Hugging Face Transformers (`Trainer`) · scikit-learn · pandas · NumPy · Matplotlib · KaggleHub · Weights & Biases · Gradio (Hugging Face Spaces)

---

## Topics

`dlgenai` · `emotion-classification` · `emotion-classifier` · `iitm` · `iitm-bs` · `iitmadrasonlinedegree` · `iitmbsc`

---

## License

This project is for educational and academic purposes under the DLGenAI course (September 2025 term), IIT Madras BS Degree in Data Science and Applications.

---

## Acknowledgements

- IIT Madras BS Data Science program
- Kaggle datasets and runtime environment
- Hugging Face Transformers library and Spaces
- PyTorch team

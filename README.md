# DLGenAI-Project-t32025

Deep Learning and GenAI Project for the September 2025 term  
Course Code: DLGenAI — IIT Madras BSc Degree in Data Science and Applications

---

## 📌 Project Title

**Multi-Label Emotion Classification using Scratch Neural Networks and Transformer Models**

This project implements a complete, modular deep learning pipeline for detecting five emotions in text:

- anger  
- fear  
- joy  
- sadness  
- surprise  

The system includes:

- A scratch-built BiLSTM model  
- Transformer-based models (BERT and RoBERTa)  
- Unified training and inference scripts  
- Model uploading and reporting utilities  
- Kaggle-ready and GitHub-ready structure (compatible with the standard Kaggle Python image)[attached_file:1]  

---

## 👤 Author

**Gaurav Tomar**  
*Student ID:* 23f1000805  
*Program:* BSc Degree in Data Science and Applications — IIT Madras  

---

## 📁 Folder Structure

emotion-classification-project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/ # Not uploaded to GitHub
│ ├── train.csv
│ ├── test.csv
│ └── sample_submission.csv
│
├── scripts/
│ ├── config.py # Paths, constants, device setup, W&B settings
│ ├── data_loader.py # Loads train/test CSV, performs train/val split
│ ├── vocab_scratch.py # Vocabulary building + text encoding for BiLSTM
│ ├── scratch_model.py # Simple BiLSTM scratch model
│ ├── scratch_train.py # Training loop for scratch model
│ ├── transformer_train.py # BERT & RoBERTa training using HF Trainer
│ ├── inference_utils.py # Shared inference helpers
│ ├── uploader.py # Upload models to KaggleHub (optional)
│ ├── reporting.py # Summaries, plots, and evaluation reports
│ ├── train.py # Main script: trains scratch + transformer models
│ └── inference.py # Main inference script
│
├── notebooks/
│ ├── dl-23f1000805-notebook-t32025-train.ipynb # preprocessing + training
│ └── dl-23f1000805-notebook-t32025-infer.ipynb # inference + submission
│
└── project_outputs/ # Created at runtime (models, logs, submissions)

text

---

## 🚀 Features

### 1. Scratch BiLSTM Model

- Custom vocabulary builder  
- Embedding + BiLSTM + linear classification head  
- Trains fully from scratch on tokenized text  
- Helps understand NLP pipelines without pretrained models  

### 2. Transformer-Based Models

- BERT-base (`bert-base-uncased`)  
- RoBERTa-base (`roberta-base`)  
- Uses Hugging Face `Trainer` API  
- Multi-label classification with sigmoid activation and BCE-with-logits loss  

### 3. Unified Training

- `train.py` runs:  
  - scratch BiLSTM training  
  - BERT fine-tuning  
  - RoBERTa fine-tuning  
- Saves checkpoints and submissions to `project_outputs/`  

### 4. Unified Inference

- `inference.py` loads:  
  - scratch model state  
  - fine-tuned BERT model  
  - fine-tuned RoBERTa model  
- Generates Kaggle-ready CSVs:  
  - `submission_scratch.csv`  
  - `submission_bert.csv`  
  - `submission_roberta.csv`  

### 5. Reporting Tools

- CSV summary of model scores  
- Bar chart and line plot of F1 scores  
- Simple text-based observations via `reporting.py`  

### 6. KaggleHub Upload Support (Optional)

- Upload complete Hugging Face model folders  
- Upload individual `.pt` files (state dicts)  
- Automatic versioning via KaggleHub  

---

## 🛠 Setup Instructions

### 1. Install Dependencies

pip install -r requirements.txt

text

Required libraries include:

- torch  
- transformers  
- scikit-learn  
- pandas  
- numpy  
- matplotlib  

---

## 📌 How to Run (Local or Kaggle)

### Step 1 — Place Dataset

Place the dataset in:

data/
train.csv
test.csv
sample_submission.csv

text

### Step 2 — Run Training

python scripts/train.py

text

This will:

- Load dataset  
- Train scratch BiLSTM  
- Train BERT  
- Train RoBERTa  
- Save models and submissions in `project_outputs/`  

### Step 3 — Run Inference

python scripts/inference.py

text

This creates:

- `submission_scratch.csv`  
- `submission_bert.csv`  
- `submission_roberta.csv`  

---

## 📌 How to Use on Kaggle

### Step 1 — Upload the project (ZIP or via Git)

Upload a ZIP of the repo or connect the GitHub repo as a Kaggle Dataset.  

### Step 2 — Extract into a folder

!unzip project.zip -d project

text

### Step 3 — Copy Kaggle dataset

!cp /kaggle/input/2025-sep-dl-gen-ai-project/* project/data/

text

### Step 4 — Run training

!python project/scripts/train.py

text

### Step 5 — Run inference

!python project/scripts/inference.py

text

---

## 📊 Model Output Summary (Expected)

| Model           | Type                | Expected F1-Micro |
|----------------|---------------------|-------------------|
| Scratch BiLSTM | From-scratch model  | ~0.30–0.60        |
| BERT-base      | Transformer         | ~0.84–0.86        |
| RoBERTa-base   | Transformer         | ~0.85–0.87        |

Actual scores depend on hyperparameters, random seed, and validation split.

---

## 📤 Uploading Models via KaggleHub (Optional)

Use `scripts/uploader.py` to upload:

- Hugging Face model folders  
- `.pt` files  
- Scratch model weights  

Example:

from scripts.uploader import upload_folder_to_kagglehub

upload_folder_to_kagglehub(
local_dir="project_outputs/bert_base_hf",
model_slug="username/emotion-models",
framework="pytorch",
notes="BERT-base fine-tuned on DLGenAI emotion dataset",
)

text

---

## 📈 Reporting

The script `scripts/reporting.py` provides:

- CSV summary of model scores  
- Bar plot of F1 scores  
- Line plot comparison across models  
- Text summary of observations  

Example:

from scripts.reporting import save_summary, plot_bar, plot_line, print_observations

df, summary_path = save_summary(results, "project_outputs")
plot_bar(df)
plot_line(df)
print_observations(df)

text

---

## 🔧 Technologies Used

- Python  
- PyTorch  
- Hugging Face Transformers  
- scikit-learn  
- pandas / numpy  
- matplotlib  
- KaggleHub (optional)  
- Weights & Biases (optional)  

---

## 📚 Learning Outcomes

- Built a complete NLP pipeline from scratch  
- Understood multi-label text classification  
- Compared scratch models vs. transformer-based models  
- Learned Hugging Face `Trainer` workflow  
- Managed a modular ML project (training, inference, reporting, upload)  
- Created Kaggle-ready and GitHub-ready ML scripts  
- Implemented unified inference for deployment-style use  

---

## 📄 License

This project is for educational and academic purposes under the DLGenAI course (September 2025 term).

---

## 📝 Acknowledgements

- IIT Madras BSc Program  
- Kaggle datasets and runtime environment[attached_file:1]  
- Hugging Face Transformers library  
- PyTorch team 

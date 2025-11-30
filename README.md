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
- Kaggle-ready and GitHub-ready structure  

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
│ └── reporting.py # Summaries, plots, and evaluation reports
│ └── train.py # Main script: trains scratch + transformer models
│ └── inference.py # Main inference script
│
├── scripts/
│ ├── dl-23f1000805-notebook-t32025 (1).ipynb # model preprocessing and training notebook
│ └── dl-23f1000805-notebook-t32025 (2).ipynb # model inference notebook
└── README.md

markdown
Copy code

---

## 🚀 Features

### **1. Scratch BiLSTM Model**
- Custom vocabulary builder  
- Embedding + BiLSTM + Linear head  
- Trains fully from scratch  
- Helps understand NLP pipelines without pretrained models  

### **2. Transformer-Based Models**
- BERT-base (bert-base-uncased)  
- RoBERTa-base (roberta-base)  
- Uses HuggingFace Trainer  
- Multi-label classification via sigmoid activation  

### **3. Unified Training**
- `train.py` runs:
  - scratch model training  
  - BERT fine-tuning  
  - RoBERTa fine-tuning  
- Saves outputs to `project_outputs/`

### **4. Unified Inference**
- `inference.py` loads:
  - scratch model state  
  - fine-tuned BERT model  
  - fine-tuned RoBERTa model  
- Generates:
  - submission_scratch.csv  
  - submission_bert.csv  
  - submission_roberta.csv  

### **5. Reporting Tools**
- Summary table  
- Bar chart  
- Line plot  
- Simple text-based observations  

### **6. KaggleHub Upload Support**
- Upload complete HuggingFace model folders  
- Upload individual `.pt` files  
- Automatically versions models  

---

## 🛠 Setup Instructions

### **1. Install Dependencies**

Use:

pip install -r requirements.txt

yaml
Copy code

Required libraries include:
- torch  
- transformers  
- scikit-learn  
- pandas  
- numpy  
- matplotlib  

---

## 📌 How to Run (Local or Kaggle)

### **Step 1 — Place Dataset**

Place the dataset in:

data/
train.csv
test.csv
sample_submission.csv

markdown
Copy code

### **Step 2 — Run Training**

python train.py

markdown
Copy code

This will:

- Load dataset  
- Train scratch BiLSTM  
- Train BERT  
- Train RoBERTa  
- Save models + submissions in:  
  `project_outputs/`

### **Step 3 — Run Inference**

python inference.py

yaml
Copy code

This creates:

- submission_scratch.csv  
- submission_bert.csv  
- submission_roberta.csv  

---

## 📌 How to Use on Kaggle

### **Step 1 — Upload the project (ZIP or via scripts)**  
### **Step 2 — Extract into a folder:**

!unzip project.zip -d project

markdown
Copy code

### **Step 3 — Copy Kaggle dataset:**

!cp /kaggle/input/2025-sep-dl-gen-ai-project/* project/data/

markdown
Copy code

### **Step 4 — Run training:**

!python project/train.py

markdown
Copy code

### **Step 5 — Run inference:**

!python project/inference.py

yaml
Copy code

---

## 📊 Model Output Summary (Expected)

| Model | Type | Expected F1-Micro |
|-------|-------|-------------------|
| Scratch BiLSTM | From-scratch model | ~0.30–0.60 |
| BERT-base | Transformer | ~0.84–0.86 |
| RoBERTa-base | Transformer | ~0.85–0.87 |

Actual scores depend on hyperparameters and random seed.

---

## 📤 Uploading Models via KaggleHub (Optional)

Use `src/uploader.py` to upload:

- HuggingFace model folders  
- `.pt` files  
- scratch model weights  

Example:

```python
from src.uploader import upload_folder_kagglehub
upload_folder_kagglehub("project_outputs/bert_base_hf", "username/emotion-models")
📈 Reporting
The script src/reporting.py provides:

CSV summary of model scores

Bar plot of F1 scores

Line plot comparison

Text summary of observations

Example:

python
Copy code
from src.reporting import *
df, _ = save_summary(results, "project_outputs")
plot_bar(df)
plot_line(df)
print_observations(df)
🔧 Technologies Used
Python

PyTorch

HuggingFace Transformers

scikit-learn

pandas / numpy

matplotlib

KaggleHub (optional)

Weights & Biases (optional)

📚 Learning Outcomes
Built a complete NLP pipeline from scratch

Understood multi-label classification

Compared scratch models vs transformers

Learned HuggingFace Trainer workflow

Managed a real project with modular code

Created Kaggle-ready & GitHub-ready ML scripts

Implemented unified inference for deployment

📄 License
This project is for educational and academic purposes under the DLGenAI course (September 2025).

📝 Acknowledgements
IIT Madras BSc Program

Kaggle datasets and runtime environment

HuggingFace Transformers library

PyTorch team

End of README

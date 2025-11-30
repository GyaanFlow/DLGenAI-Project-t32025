import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.vocab_scratch import build_vocab, encode_text
from src.scratch_model import SimpleBiLSTM
from src.config import DEVICE, EMOTION_LABELS

DATA_DIR = "/kaggle/input/2025-sep-dl-gen-ai-project"
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

# SCRATCH_FOLDER = "/kaggle/input/.../1"
# BERT_FOLDER = "/kaggle/input/.../1"
# ROBERTA_FOLDER = "/kaggle/input/.../1"
SCRATCH_FOLDER = "/kaggle/input/emotion-models-20251129-202524/pytorch/emotion-models-20251129-202524-20251129-202524/1"
BERT_FOLDER    = "/kaggle/input/emotion-models-20251129-202524/pytorch/emotion-models-20251129-202524-20251129-202529/1"
ROBERTA_FOLDER = "/kaggle/input/emotion-models-20251129-202524/pytorch/emotion-models-20251129-202524-20251129-202542/1"

# Scratch
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
vocab = build_vocab(train_df["text"], 10000)
pad_idx = vocab["<pad>"]

model = SimpleBiLSTM(len(vocab), 128, 128, pad_idx=pad_idx).to(DEVICE)
state = torch.load(f"{SCRATCH_FOLDER}/best_scratch_state.pt", map_location=DEVICE)
model.load_state_dict(state)
model.eval()

inputs = torch.tensor([encode_text(t, vocab, 50) for t in test_df["text"]], dtype=torch.long).to(DEVICE)
with torch.no_grad():
    probs = torch.sigmoid(model(inputs)).cpu().numpy()
    scratch_preds = (probs > 0.5).astype(int)

# BERT
tok = AutoTokenizer.from_pretrained(BERT_FOLDER)
mdl = AutoModelForSequenceClassification.from_pretrained(BERT_FOLDER).to(DEVICE)
mdl.eval()

enc = tok(list(test_df["text"]), padding="max_length", truncation=True, max_length=128, return_tensors="pt")
enc = {k: v.to(DEVICE) for k,v in enc.items()}
with torch.no_grad():
    probs = torch.sigmoid(mdl(**enc).logits).cpu().numpy()
bert_preds = (probs > 0.5).astype(int)

# RoBERTa
tok = AutoTokenizer.from_pretrained(ROBERTA_FOLDER)
mdl = AutoModelForSequenceClassification.from_pretrained(ROBERTA_FOLDER).to(DEVICE)
mdl.eval()

enc = tok(list(test_df["text"]), padding="max_length", truncation=True, max_length=128, return_tensors="pt")
enc = {k: v.to(DEVICE) for k,v in enc.items()}
with torch.no_grad():
    probs = torch.sigmoid(mdl(**enc).logits).cpu().numpy()
roberta_preds = (probs > 0.5).astype(int)

print("Inference completed.")

from pathlib import Path
import pandas as pd

from src.config import OUT_DIR
from src.data_loader import load_raw, train_val_split
from src.scratch_train import train_scratch
from src.transformer_train import train_transformer
from src.config import EMOTION_LABELS

DATA_DIR = "/kaggle/input/2025-sep-dl-gen-ai-project"

train_df, test_df, sample_sub = load_raw(DATA_DIR)
train_df, val_df, y_train, y_val = train_val_split(train_df)

# 1) Scratch
vocab, scratch_preds, scratch_f1, scratch_dir = train_scratch(
    train_df, val_df, test_df, y_train, y_val
)

# 2) BERT
res_bert = train_transformer(
    "bert-base-uncased",
    train_df["text"],
    y_train,
    val_df["text"],
    y_val,
    test_df["text"],
    "bert_base"
)

# 3) RoBERTa
res_roberta = train_transformer(
    "roberta-base",
    train_df["text"],
    y_train,
    val_df["text"],
    y_val,
    test_df["text"],
    "roberta_base"
)

print("Training complete.")

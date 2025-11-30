import numpy as np
import torch
import shutil
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)

from src.config import DEVICE, OUT_DIR, USE_WANDB, EMOTION_LABELS

class HFDS(Dataset):
    def __init__(self, enc, labels=None):
        self.enc = enc
        self.labels = labels

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

def train_transformer(model_name, train_texts, train_labels, val_texts, val_labels, test_texts, model_tag):

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    MAXLEN = 128

    def encode(X):
        return tokenizer(list(X), padding="max_length", truncation=True, max_length=MAXLEN)

    train_enc = encode(train_texts)
    val_enc = encode(val_texts)
    test_enc = encode(test_texts)

    train_ds = HFDS(train_enc, train_labels)
    val_ds = HFDS(val_enc, val_labels)
    test_ds = HFDS(test_enc, np.zeros((len(test_texts), len(EMOTION_LABELS))))

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(EMOTION_LABELS),
        problem_type="multi_label_classification"
    ).to(DEVICE)

    args = TrainingArguments(
        output_dir=str(OUT_DIR / f"{model_tag}_tmp"),
        num_train_epochs=8,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_ratio=0.1,
        weight_decay=0.02,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="wandb" if USE_WANDB else None
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs > 0.5).astype(int)
        return {
            "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0)
        }

    if USE_WANDB:
        import wandb
        wandb.init(project="23f1000805-t32025", name=model_tag, reinit=True)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()

    # evaluation
    val_pred = trainer.predict(val_ds)
    val_metrics = compute_metrics((val_pred.predictions, val_pred.label_ids))

    # test predictions
    test_pred = trainer.predict(test_ds)
    probs = torch.sigmoid(torch.tensor(test_pred.predictions)).numpy()
    test_bin = (probs > 0.5).astype(int)

    # submission
    sub = pd.DataFrame(test_bin, columns=EMOTION_LABELS)
    sub.insert(0, "id", range(len(test_bin)))
    sub_path = OUT_DIR / f"submission_{model_tag}.csv"
    sub.to_csv(sub_path, index=False)

    # save HF
    hf_dir = OUT_DIR / f"{model_tag}_hf"
    trainer.save_model(str(hf_dir))
    tokenizer.save_pretrained(str(hf_dir))

    return {
        "model_tag": model_tag,
        "val_metrics": val_metrics,
        "submission": str(sub_path),
        "hf_folder": str(hf_dir)
    }

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from src.config import EMOTION_LABELS

def load_raw(DATA_DIR):
    DATA_DIR = Path(DATA_DIR)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return train_df, test_df, sample_sub

def train_val_split(train_df, test_size=0.2):
    train_df, val_df = train_test_split(train_df, test_size=test_size, random_state=42)
    y_train = train_df[EMOTION_LABELS].values
    y_val = val_df[EMOTION_LABELS].values
    return train_df, val_df, y_train, y_val

# reporting.py
"""
Reporting utilities for summarizing model performance.
Creates CSV summaries, simple charts, and prints observations.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def save_summary(results, out_dir):
    """
    Take a list of result dicts and save a summary CSV.
    Each dict should contain:
        - model
        - f1_micro
        - submission
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    summary_path = out_dir / "model_summary.csv"
    df.to_csv(summary_path, index=False)

    print("Saved summary to:", summary_path)
    print(df)
    return df, summary_path


def plot_bar(df):
    """
    Bar chart of F1 scores for all models.
    """
    plt.figure(figsize=(7, 4))
    plt.bar(df["model"], df["f1_micro"])
    plt.title("F1 Micro Score by Model")
    plt.ylabel("F1 Micro")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.show()


def plot_line(df):
    """
    Line plot of model performance.
    """
    plt.figure(figsize=(7, 4))
    plt.plot(df["model"], df["f1_micro"], marker="o")
    plt.title("Model Comparison (F1 Micro)")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xticks(rotation=30)
    plt.show()


def print_observations(df):
    """
    Print a simple text-based report of all model results.
    """
    print("\nObservations:")
    for _, row in df.iterrows():
        print(f"- {row['model']}: F1 Micro = {row['f1_micro']:.4f}, File = {row['submission']}")

    print("\nGeneral Notes:")
    print("- Transformer models perform better than scratch models.")
    print("- Scratch model is useful for understanding the basic pipeline.")
    print("- BERT/RoBERTa usually achieve strong scores with fewer epochs.")
    print("- The best model should be used for deployment or inference.")

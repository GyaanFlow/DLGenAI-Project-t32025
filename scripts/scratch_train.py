import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import shutil

from src.config import DEVICE, OUT_DIR, USE_WANDB
from src.scratch_model import SimpleBiLSTM
from src.vocab_scratch import SimpleDataset, build_vocab

def train_scratch(train_df, val_df, test_df, y_train, y_val, max_len=50):

    vocab = build_vocab(train_df["text"].tolist(), max_words=10000)
    vocab_size = len(vocab)

    train_ds = SimpleDataset(train_df["text"].tolist(), y_train, vocab, max_len)
    val_ds = SimpleDataset(val_df["text"].tolist(), y_val, vocab, max_len)
    test_ds = SimpleDataset(test_df["text"].tolist(), None, vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    model = SimpleBiLSTM(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=128,
        pad_idx=vocab["<pad>"]
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    if USE_WANDB:
        import wandb
        wandb.init(project="23f1000805-t32025", name="scratch-bilstm", reinit=True)

    best_f1 = 0
    out_dir = OUT_DIR / "scratch_model"
    out_dir.mkdir(exist_ok=True)

    for ep in range(1, 101):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # validation
        model.eval()
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                probs = torch.sigmoid(model(xb)).cpu().numpy()
                preds_all.append((probs > 0.5).astype(int))
                labels_all.append(yb.numpy())

        preds_all = np.vstack(preds_all)
        labels_all = np.vstack(labels_all)
        f1 = f1_score(labels_all, preds_all, average="micro", zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), out_dir / "best_scratch_state.pt")
            torch.save(model, out_dir / "best_scratch_full.pt")

        print(f"[Scratch] Epoch {ep} F1={f1:.4f}")

    if USE_WANDB:
        wandb.finish()

    # generate test predictions
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            preds.append((probs > 0.5).astype(int))

    return vocab, np.vstack(preds), best_f1, out_dir

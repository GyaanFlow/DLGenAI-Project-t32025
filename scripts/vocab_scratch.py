import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
from src.config import EMOTION_LABELS

def build_vocab(texts, max_words=10000):
    c = Counter()
    for t in texts:
        c.update(t.lower().split())
    vocab = {"<pad>":0, "<unk>":1}
    for i, (w, _) in enumerate(c.most_common(max_words)):
        vocab[w] = i + 2
    return vocab

def encode_text(text, vocab, max_len=50):
    words = text.lower().split()
    ids = [vocab.get(w, vocab["<unk>"]) for w in words]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

class SimpleDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = encode_text(self.texts[idx], self.vocab, self.max_len)
        y = self.labels[idx] if self.labels is not None else np.zeros(len(EMOTION_LABELS))
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float32)

import torch
import torch.nn as nn
from src.config import EMOTION_LABELS

class SimpleBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, len(EMOTION_LABELS))

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        return self.fc(pooled)

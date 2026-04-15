"""
model.py — Bidirectional GRU sentiment classifier
"""

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """
    Embedding → Bidirectional GRU → concat(fwd, bwd final states) → Linear

    Architecture notes:
      - GRU gates (reset + update) solve vanishing gradients vs vanilla RNN
      - Bidirectional: one pass left→right, one right→left; final hidden states
        are concatenated, giving the classifier context from both directions
      - Output is raw logits (no softmax); CrossEntropyLoss handles that internally
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 num_classes, dropout, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb       = self.dropout(self.embedding(x))       # (B, L, E)
        _, hidden = self.gru(emb)                         # hidden: (2*layers, B, H)
        # Last layer forward + backward final states
        last = torch.cat([hidden[-2], hidden[-1]], dim=1) # (B, 2H)
        return self.fc(self.dropout(last))                # (B, num_classes)

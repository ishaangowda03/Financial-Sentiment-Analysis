"""
evaluate.py — Evaluate trained model on the held-out test set

Usage:
    python evaluate.py

Requires:
    best_fpb.pt and vocab_config.pt (produced by train.py)
"""

import torch
from torch.utils.data import DataLoader
from collections import Counter

from data import load_fpb, stratified_split, FPBDataset, LABEL_NAMES
from model import GRUClassifier

SEED       = 42
BATCH_SIZE = 32
MODEL_PATH = "best_fpb.pt"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load vocab + config saved during training ──────────────────────────────────
checkpoint = torch.load("vocab_config.pt", weights_only=False)
vocab  = checkpoint["vocab"]
cfg    = checkpoint["config"]

# ── Rebuild test split (same seed = same split as training) ────────────────────
all_data = load_fpb("sentences_allagree")
_, _, test_data = stratified_split(all_data, seed=SEED)
test_loader = DataLoader(
    FPBDataset(test_data, vocab, cfg["max_len"]),
    batch_size=BATCH_SIZE,
)

# ── Load model ─────────────────────────────────────────────────────────────────
model = GRUClassifier(
    len(vocab), cfg["embed_dim"], cfg["hidden_dim"],
    cfg["num_layers"], cfg["num_classes"], cfg["dropout"],
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# ── Collect predictions ────────────────────────────────────────────────────────
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb.to(DEVICE))
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(yb.tolist())

n = len(all_labels)
overall_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n

# ── Per-class metrics ──────────────────────────────────────────────────────────
print(f"\n── Test Results (n={n}) ──────────────────────────────────────────────")
print(f"  Overall Accuracy : {overall_acc:.2%}\n")
print(f"  {'Class':>10} | {'Precision':>9} | {'Recall':>7} | {'F1':>6} | {'Support':>7}")
print(f"  {'─'*10}─┼─{'─'*9}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*7}")

f1_scores = []
for cls in range(cfg["num_classes"]):
    tp   = sum(p == cls and l == cls for p, l in zip(all_preds, all_labels))
    fp   = sum(p == cls and l != cls for p, l in zip(all_preds, all_labels))
    fn   = sum(p != cls and l == cls for p, l in zip(all_preds, all_labels))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    supp = sum(l == cls for l in all_labels)
    f1_scores.append(f1)
    print(f"  {LABEL_NAMES[cls]:>10} | {prec:>9.2%} | {rec:>7.2%} | {f1:>6.2%} | {supp:>7}")

macro_f1 = sum(f1_scores) / len(f1_scores)
print(f"\n  Macro F1: {macro_f1:.2%}")
print(f"\n  (FinBERT reference: ~88% macro F1 on this dataset)")

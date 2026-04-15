"""
train.py — Train the GRU sentiment classifier on Financial PhraseBank

Usage:
    python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter

from data import load_fpb, build_vocab, stratified_split, FPBDataset, LABEL_NAMES
from model import GRUClassifier

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEED        = 42
EMBED_DIM   = 128
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.4
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-3
MAX_LEN     = 40
MIN_FREQ    = 2
NUM_CLASSES = 3
MODEL_PATH  = "best_fpb.pt"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
print(f"Device: {DEVICE}\n")

# ── Data ───────────────────────────────────────────────────────────────────────
print("Loading Financial PhraseBank (sentences_allagree)...")
all_data = load_fpb("sentences_allagree")

label_counts = Counter(l for _, l in all_data)
print(f"Total samples: {len(all_data)}")
for k, v in sorted(label_counts.items()):
    print(f"  {LABEL_NAMES[k]:>8}: {v}")
print()

train_data, val_data, test_data = stratified_split(all_data, seed=SEED)
print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

vocab      = build_vocab(train_data, min_freq=MIN_FREQ)  # vocab built on train only
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}\n")

train_loader = DataLoader(FPBDataset(train_data, vocab, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(FPBDataset(val_data,   vocab, MAX_LEN), batch_size=BATCH_SIZE)

# Save vocab and config for inference
torch.save({"vocab": vocab, "config": {
    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
    "num_layers": NUM_LAYERS, "dropout": DROPOUT,
    "num_classes": NUM_CLASSES, "max_len": MAX_LEN,
}}, "vocab_config.pt")

# ── Model ──────────────────────────────────────────────────────────────────────
model = GRUClassifier(
    vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT
).to(DEVICE)

# Class weights: inverse frequency — no normalization, just relative penalties
counts  = [label_counts[i] for i in range(NUM_CLASSES)]
weights = torch.tensor([1.0 / c for c in counts], dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print(f"Trainable parameters: {total_params:,}\n")

# ── Training Loop ──────────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot_loss = tot_correct = tot_n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss   = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tot_loss    += loss.item() * len(yb)
            tot_correct += (logits.argmax(1) == yb).sum().item()
            tot_n       += len(yb)
    return tot_loss / tot_n, tot_correct / tot_n

print(f"{'Ep':>3} | {'Tr Loss':>8} | {'Tr Acc':>7} | {'Val Loss':>9} | {'Val Acc':>8} | {'LR':>8}")
print("─" * 62)

best_val_acc = 0.0
for ep in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)
    scheduler.step(vl_loss)
    lr = optimizer.param_groups[0]["lr"]

    marker = " ◀ best" if vl_acc > best_val_acc else ""
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), MODEL_PATH)

    print(f"{ep:>3} | {tr_loss:>8.4f} | {tr_acc:>7.4f} | {vl_loss:>9.4f} | {vl_acc:>8.4f} | {lr:>8.2e}{marker}")

print(f"\nBest val accuracy: {best_val_acc:.2%}")
print(f"Model saved to {MODEL_PATH}")
print("\nRun evaluate.py to see test results.")

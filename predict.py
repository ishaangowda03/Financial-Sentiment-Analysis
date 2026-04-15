"""
predict.py — Run inference on custom financial headlines

Usage:
    python predict.py
    python predict.py --text "Company revenue fell 15 percent amid weakening demand"

Requires:
    best_fpb.pt and vocab_config.pt (produced by train.py)
"""

import argparse
import torch

from data import encode, LABEL_NAMES
from model import GRUClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load vocab + model ─────────────────────────────────────────────────────────
checkpoint = torch.load("vocab_config.pt", weights_only=False)
vocab = checkpoint["vocab"]
cfg   = checkpoint["config"]

model = GRUClassifier(
    len(vocab), cfg["embed_dim"], cfg["hidden_dim"],
    cfg["num_layers"], cfg["num_classes"], cfg["dropout"],
).to(DEVICE)
model.load_state_dict(torch.load("best_fpb.pt", map_location=DEVICE, weights_only=True))
model.eval()

ICONS = {0: "📉 NEGATIVE", 1: "➡️  NEUTRAL ", 2: "📈 POSITIVE"}


def predict(text: str):
    with torch.no_grad():
        ids    = torch.tensor([encode(text, vocab, cfg["max_len"])], dtype=torch.long).to(DEVICE)
        logits = model(ids)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()
    return ICONS[pred], probs[pred].item(), {LABEL_NAMES[i]: f"{probs[i].item():.1%}" for i in range(3)}


def run_demo():
    headlines = [
        "The company reported record quarterly earnings exceeding analyst expectations",
        "Operating profit fell 23 percent as raw material costs surged unexpectedly",
        "Nokia said it will supply telecom infrastructure to a major European carrier",
        "The board approved a share buyback programme worth 500 million euros",
        "Revenue was in line with forecasts as demand remained broadly stable",
        "The firm announced restructuring plans affecting approximately 1200 employees",
    ]
    print("\n── Inference Demo ───────────────────────────────────────────────────────")
    for h in headlines:
        label, conf, breakdown = predict(h)
        print(f"  {label} ({conf:.1%})  →  {h}")
        print(f"             probs: {breakdown}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None,
                        help="Single headline to classify")
    args = parser.parse_args()

    if args.text:
        label, conf, breakdown = predict(args.text)
        print(f"\n  {label} ({conf:.1%})")
        print(f"  probs: {breakdown}\n")
    else:
        run_demo()

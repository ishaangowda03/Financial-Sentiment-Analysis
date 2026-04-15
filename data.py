"""
data.py — Data loading, vocabulary, encoding, and stratified splits
"""

import re
import random
from collections import Counter

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split

PAD, UNK = "<PAD>", "<UNK>"
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def load_fpb(config="sentences_allagree"):
    """Load Financial PhraseBank and return list of (sentence, label) tuples."""
    ds = load_dataset("takala/financial_phrasebank", config, trust_remote_code=True)
    return [(row["sentence"], row["label"]) for row in ds["train"]]


def tokenize(text):
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def build_vocab(data, min_freq=2):
    counts = Counter(tok for text, _ in data for tok in tokenize(text))
    vocab  = {PAD: 0, UNK: 1}
    for word, freq in counts.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_len):
    ids  = [vocab.get(t, vocab[UNK]) for t in tokenize(text)[:max_len]]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


def stratified_split(data, val_size=0.15, test_size=0.15, seed=42):
    """
    Stratified train / val / test split.
    Preserves class distribution across all three splits — important for
    imbalanced datasets like FPB where neutral is ~60% of samples.
    """
    texts  = [t for t, _ in data]
    labels = [l for _, l in data]

    # First split off test set
    texts_tv, texts_test, labels_tv, labels_test = train_test_split(
        texts, labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )
    # Then split val from remaining train
    relative_val = val_size / (1.0 - test_size)
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts_tv, labels_tv,
        test_size=relative_val,
        stratify=labels_tv,
        random_state=seed,
    )

    train = list(zip(texts_train, labels_train))
    val   = list(zip(texts_val,   labels_val))
    test  = list(zip(texts_test,  labels_test))
    return train, val, test


class FPBDataset(Dataset):
    def __init__(self, data, vocab, max_len):
        self.samples = [
            (
                torch.tensor(encode(t, vocab, max_len), dtype=torch.long),
                torch.tensor(l, dtype=torch.long),
            )
            for t, l in data
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

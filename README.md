# Financial Sentiment Analysis — Bidirectional GRU

3-class sentiment classifier (positive / neutral / negative) trained on real financial news headlines using a bidirectional GRU built from scratch in PyTorch.

---

## Dataset

**Financial PhraseBank** — [`takala/financial_phrasebank`](https://huggingface.co/datasets/takala/financial_phrasebank)

- ~2,264 sentences from English financial news (OMX Helsinki)
- Annotated by 16 people with finance/business backgrounds
- Labels reflect investor perspective: will this news move a stock up, down, or neither?
- Using `sentences_allagree` config: only sentences where all annotators agreed → cleanest labels

```
negative:  ~10%
neutral:   ~60%
positive:  ~30%
```

---

## Architecture

```
Input tokens
    ↓
Embedding layer (128-dim)
    ↓
Bidirectional GRU (256 hidden, 2 layers)
    ↓
Concat(forward final state, backward final state)  →  512-dim
    ↓
Dropout (0.4)
    ↓
Linear → 3 logits  →  CrossEntropyLoss
```

**Why GRU over vanilla RNN?**  
GRU's reset and update gates control how much past context to retain, which solves the vanishing gradient problem that makes vanilla RNNs fail on longer sequences.

**Why bidirectional?**  
Financial sentences often have sentiment-bearing words near the end ("...fell sharply" or "...beat expectations"). Running a second GRU right-to-left ensures those words inform the full representation.

**Why class weights?**  
Neutral is ~60% of the dataset. Without reweighting, the model learns to predict neutral for everything and still gets decent accuracy. Inverse-frequency weights penalize misclassifying the minority classes more heavily.

---

## Results

| Class    | Precision | Recall | F1    |
|----------|-----------|--------|-------|
| negative | ~72%      | ~68%   | ~70%  |
| neutral  | ~78%      | ~82%   | ~80%  |
| positive | ~74%      | ~71%   | ~72%  |
| **Macro F1** |       |        | **~74%** |

> Results are approximate — vary slightly across runs due to random initialization.  
> **FinBERT reference**: ~88% macro F1 on the same dataset.  
> The ~14-point gap reflects the fundamental difference between a small scratch-trained RNN and a transformer pre-trained on billions of financial tokens.

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/rnn-financial-sentiment
cd rnn-financial-sentiment
pip install -r requirements.txt

# Train (saves best_fpb.pt + vocab_config.pt)
python train.py

# Evaluate on test set
python evaluate.py

# Classify a custom headline
python predict.py --text "Company revenue fell 15 percent amid weakening demand"

# Run inference demo
python predict.py
```

---

## Project Structure

```
rnn-financial-sentiment/
├── model.py          # GRUClassifier architecture
├── data.py           # Loading, tokenization, vocab, stratified splits
├── train.py          # Training loop, checkpointing
├── evaluate.py       # Test set evaluation — per-class + macro F1
├── predict.py        # Inference on custom text
├── requirements.txt
└── .gitignore
```

---

## Citation

```bibtex
@article{Malo2014GoodDO,
  title   = {Good debt or bad debt: Detecting semantic orientations in economic texts},
  author  = {P. Malo and A. Sinha and P. Korhonen and J. Wallenius and P. Takala},
  journal = {Journal of the Association for Information Science and Technology},
  year    = {2014},
  volume  = {65}
}
```

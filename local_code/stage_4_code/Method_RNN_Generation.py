"""
RNN (LSTM) text generation on the jokes dataset.
Trains on next-word prediction, generates from 3 starting words,
and compares generations to the training data via n-gram overlap.

Run:  python rnn_jokes_generation.py
"""
import os, re, math, random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------- CONFIG (edit these) -------------------------
DATA_PATH   = "../../stage_4_data/text_generation/data"   # file (or folder) with jokes
RESULT_DIR  = "../../result/stage_4_result/RNN_Generation/"
SEQ_LEN     = 30      # window length (tokens fed to the model)
STRIDE      = 3       # step between windows (bigger = faster, less data)
EMB_DIM     = 128
HIDDEN      = 256
NUM_LAYERS  = 2
DROPOUT     = 0.3
BATCH_SIZE  = 128
EPOCHS      = 15
LR          = 1e-3
MIN_FREQ    = 2       # words rarer than this become <unk>
TEMPERATURE = 0.8
PROMPTS     = ["what do you", "why did the", "i told my"]
SEED        = 2
# -----------------------------------------------------------------------

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"


# ---------------------------- data ----------------------------
def tokenize(text):
    # words (with apostrophes) and punctuation as separate tokens
    return re.findall(r"[a-z0-9']+|[^\sa-z0-9]", text.lower())


def load_jokes(path):
    files = [path]
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in sorted(os.listdir(path))
                 if f.lower().endswith((".txt", ".csv")) or "." not in f]
    jokes = []
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
            for i, line in enumerate(fh):
                line = line.strip().strip('"')
                if not line:
                    continue
                if i == 0 and line.lower().startswith(("id,", "id\t", "joke")):
                    continue  # header row
                toks = tokenize(line)
                if toks:
                    jokes.append(toks)
    return jokes


def build_vocab(jokes, min_freq):
    counter = Counter(t for j in jokes for t in j)
    vocab = {PAD: 0, UNK: 1, EOS: 2}
    for tok, c in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def make_windows(jokes, vocab, seq_len, stride):
    stream = []
    for j in jokes:
        stream += [vocab.get(t, vocab[UNK]) for t in j] + [vocab[EOS]]
    xs, ys = [], []
    for i in range(0, len(stream) - seq_len - 1, stride):
        xs.append(stream[i:i + seq_len])
        ys.append(stream[i + 1:i + seq_len + 1])   # input shifted by one
    return torch.tensor(xs), torch.tensor(ys)


# ---------------------------- model ----------------------------
class RNNGenerator(nn.Module):
    def __init__(self, vocab_size, emb=EMB_DIM, hidden=HIDDEN,
                 layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.rnn = nn.LSTM(emb, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        h, state = self.rnn(self.drop(self.emb(x)), state)
        return self.out(self.drop(h)), state          # (B, T, vocab)


# ---------------------------- train ----------------------------
def run_epoch(model, loader, loss_fn, vocab_size, opt=None):
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = loss_fn(logits.reshape(-1, vocab_size), yb.reshape(-1))
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
    return total / n


# ---------------------------- generate ----------------------------
@torch.no_grad()
def generate(model, vocab, inv_vocab, prompt, max_len=60, temperature=TEMPERATURE):
    model.eval()
    ids = [vocab.get(t, vocab[UNK]) for t in tokenize(prompt)]
    x = torch.tensor([ids], device=device)
    logits, state = model(x)
    for _ in range(max_len):
        l = logits[0, -1] / temperature
        l[vocab[PAD]] = -1e9                      # never sample <pad>
        l[vocab[UNK]] = -1e9                      # or <unk>
        nxt = torch.multinomial(torch.softmax(l, -1), 1).item()
        if nxt == vocab[EOS]:
            break
        ids.append(nxt)
        logits, state = model(torch.tensor([[nxt]], device=device), state)
    return " ".join(inv_vocab[i] for i in ids)


# ---------------------------- evaluation ----------------------------
def ngram_set(token_lists, n):
    s = set()
    for toks in token_lists:
        s.update(zip(*[toks[i:] for i in range(n)]))
    return s


def ngram_overlap(gen_tokens, train_ngrams, n):
    grams = list(zip(*[gen_tokens[i:] for i in range(n)]))
    return sum(g in train_ngrams for g in grams) / max(len(grams), 1)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    jokes = load_jokes(DATA_PATH)
    print(f"Loaded {len(jokes)} jokes")
    random.shuffle(jokes)
    split = int(0.9 * len(jokes))
    train_jokes, val_jokes = jokes[:split], jokes[split:]

    vocab = build_vocab(train_jokes, MIN_FREQ)         # vocab from TRAIN only
    inv_vocab = {i: t for t, i in vocab.items()}
    V = len(vocab)
    print(f"Vocab size: {V} | device: {device}")

    xtr, ytr = make_windows(train_jokes, vocab, SEQ_LEN, STRIDE)
    xva, yva = make_windows(val_jokes, vocab, SEQ_LEN, STRIDE)
    train_loader = DataLoader(TensorDataset(xtr, ytr), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(xva, yva), BATCH_SIZE)
    print(f"Train windows: {len(xtr)} | Val windows: {len(xva)}")

    model = RNNGenerator(V).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab[PAD])
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_hist, val_hist = [], []
    for ep in range(1, EPOCHS + 1):
        tl = run_epoch(model, train_loader, loss_fn, V, opt)
        vl = run_epoch(model, val_loader, loss_fn, V)
        train_hist.append(tl); val_hist.append(vl)
        print(f"Epoch {ep:2d}/{EPOCHS} | train loss {tl:.4f} | "
              f"val loss {vl:.4f} | val perplexity {math.exp(vl):.2f}")

    # ---- convergence plot ----
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, EPOCHS + 1), train_hist, label="train")
    plt.plot(range(1, EPOCHS + 1), val_hist, label="validation")
    plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss")
    plt.title("RNN Generation Training Convergence"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "training_convergence.png"), dpi=150)
    plt.close()

    # ---- generate + compare with training data ----
    train_bi, train_tri = ngram_set(train_jokes, 2), ngram_set(train_jokes, 3)
    train_joke_strs = {" ".join(j) for j in train_jokes}

    lines = [f"Final val loss: {val_hist[-1]:.4f} | "
             f"val perplexity: {math.exp(val_hist[-1]):.2f}\n"]
    bi_scores, tri_scores = [], []
    print("\n************ Generated jokes ************")
    for prompt in PROMPTS:
        for _ in range(3):                         # a few samples per prompt
            text = generate(model, vocab, inv_vocab, prompt)
            toks = text.split()
            bi = ngram_overlap(toks, train_bi, 2)
            tri = ngram_overlap(toks, train_tri, 3)
            copied = text in train_joke_strs
            bi_scores.append(bi); tri_scores.append(tri)
            block = (f"[{prompt}] {text}\n   bigram overlap: {bi:.2f} | "
                     f"trigram overlap: {tri:.2f} | copied verbatim: {copied}\n")
            print(block); lines.append(block)

    summary = (f"\nAvg bigram overlap with training data:  {np.mean(bi_scores):.3f}\n"
               f"Avg trigram overlap with training data: {np.mean(tri_scores):.3f}\n")
    print(summary); lines.append(summary)

    with open(os.path.join(RESULT_DIR, "Generated_Jokes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Saved results to", RESULT_DIR)


if __name__ == "__main__":
    main()
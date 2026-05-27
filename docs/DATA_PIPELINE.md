# Sonnet — Stage 6: Data Pipeline

## What Was Built

```
sonnet_core/data.py          — Dataset, DataLoader, generators, train_test_split
examples/train_circles.py    — circles classification end-to-end
tests/test_data.py           — 17 tests
```

---

## Dataset

```python
ds = Dataset(X, y)
# X: list of lists — one inner list per sample
# y: list of floats — one target per sample

len(ds)       # number of samples
ds[i]         # (X[i], y[i])
ds[i:j]       # (X[i:j], y[i:j])
```

Thin wrapper. No transformation, no preprocessing — just indexed access. The simplicity is intentional: transformations belong in a separate pipeline stage (Stage 7 will add this).

---

## DataLoader

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for X_batch, y_batch in loader:
    # X_batch: NDTensor of shape (batch_size, n_features)
    # y_batch: NDTensor of shape (batch_size, 1)
    ...
```

Each call to `iter(loader)` produces a fresh sequence of batches. If `shuffle=True`, indices are randomized at the start of each iteration — so every epoch sees a different order.

The last batch may be smaller than `batch_size` when `n % batch_size != 0`. This is correct and expected — don't discard it.

---

## Why Mini-Batch Training Works Better

### Gradient quality

|  Mode        | Gradient estimate | Cost per step |
|------------- |-------------------|---------------|
| Full batch   | Exact             | Very high     |
| Mini-batch   | Good approximation| Moderate      |
| Single sample| Very noisy        | Very low      |

Mini-batch finds the sweet spot. A gradient over 32 samples is a statistically reasonable estimate of the true gradient, at a fraction of the cost of processing all data.

### Noise as a feature

Batch gradient noise is not purely bad. It acts as implicit regularization — the model can't perfectly memorize any single sample because it only sees it averaged with 31 others. This is one reason mini-batch SGD generalizes better than full-batch gradient descent.

### Batch size trade-offs

| Batch size | Gradient noise | Memory  | Typical use |
|-----------|----------------|---------|-------------|
| 1         | Very high      | Minimal | Online learning |
| 16–32     | Moderate       | Low     | Most tasks  |
| 64–256    | Low            | High    | Large datasets |
| Full      | None           | Maximum | Small datasets |

32 is the default for a reason: it works well across the widest range of problems.

---

## Shuffling

```python
indices = list(range(n))
random.shuffle(indices)   # in-place, O(n)
```

One shuffle at the start of each epoch. Every sample appears exactly once per epoch (no replacement, no skipping) — just in random order.

Without shuffling, if your data is sorted by class (all class-0 first, then class-1), early batches are pure class-0. The network learns a bias before it sees any class-1 examples. The shuffled gradient landscape is smoother and more representative.

---

## Train/Test Split

```python
train, test = train_test_split(dataset, test_ratio=0.2)
```

Randomly shuffles indices, then splits at `n * (1 - test_ratio)`.

The test set is held out completely — never passed to the optimizer. It measures generalization: how well the model performs on data it has never seen.

**The train/test gap is your most important diagnostic:**
- Small gap → model generalizes well
- Large gap → model is overfitting to training data

---

## Dataset Generators

### make_circles

```
Inner ring (r≈0.5): class 0
Outer ring (r≈1.0): class 1
```

Generated analytically — no file I/O, no downloads. Points sampled at uniform angles with radius perturbed by Gaussian noise. Not linearly separable: requires a curved decision boundary.

### make_moons

Two interlocking crescents. Harder than circles — the boundary is more complex and the classes are interleaved rather than separated by radius. A good stress test for network capacity.

---

## The Circles Example

Expected output:

```
epoch   1/20  loss=0.2xxx  [████░░░░░░░░░░░░░░░░]  train=65%  test=64%
epoch   5/20  loss=0.1xxx  [████████░░░░░░░░░░░░]  train=85%  test=82%
epoch  10/20  loss=0.05xx  [████████████░░░░░░░░]  train=93%  test=91%
epoch  20/20  loss=0.01xx  [████████████████████]  train=98%  test=96%
```

The decision boundary visualization shows a ring-shaped boundary — concentric, not linear:

```
░░░░░░░░░░░░░░░░░░░░░░
░░░░░░░██████░░░░░░░░░
░░░░░██████████░░░░░░░
░░░░████░░░░████░░░░░░
░░░███░░░░░░░███░░░░░░
░░░███░░░░░░░███░░░░░░
░░░░████░░░░████░░░░░░
░░░░░██████████░░░░░░░
░░░░░░░██████░░░░░░░░░
░░░░░░░░░░░░░░░░░░░░░░
```

The ring of `█` (class 1 predicted) surrounding `░` (class 0) is the visual proof that the network learned a genuinely nonlinear boundary from real data using a full training pipeline.

---

## Full Architecture

```
sonnet/
├── sonnet_core/
│   ├── tensor.py      — scalar autograd            (stage 1)
│   ├── nn.py          — scalar MLP                 (stage 2)
│   ├── loss.py        — scalar loss                (stage 3)
│   ├── trainer.py     — training loop              (stage 3)
│   ├── ndtensor.py    — matrix autograd            (stage 4)
│   ├── linear.py      — Linear, Sequential         (stage 4)
│   ├── optim.py       — SGD, Adam                  (stage 5)
│   └── data.py        — Dataset, DataLoader        (stage 6) ← new
├── examples/
│   ├── xor.py
│   └── train_circles.py                            (stage 6) ← new
└── tests/
    ├── test_backward.py
    ├── test_nn.py
    ├── test_ndtensor.py
    ├── test_optim.py
    └── test_data.py                                (stage 6) ← new
```

---

## Run

```bash
cd sonnet
python -m tests.test_data
python -m examples.train_circles
```

---

## Next Step: Stage 7 — Transformer Components

With a complete training pipeline, the next stage builds the components that define the transformer architecture:

```
Embedding          — integer token → dense vector
PositionalEncoding — inject position information
LayerNorm          — normalize across features (not batch)
MultiHeadAttention — the core of every transformer
FeedForward        — position-wise MLP inside each block
TransformerBlock   — one complete encoder/decoder layer
```

After Stage 7, Stage 8 assembles these into a character-level language model — Sonnet generates text for the first time.

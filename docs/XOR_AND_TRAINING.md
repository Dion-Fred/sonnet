# Sonnet — Loss Functions, Trainer, and XOR

## What Was Built

```
sonnet_core/loss.py      — MSELoss, BinaryCrossEntropy, binary_accuracy
sonnet_core/trainer.py   — Trainer (training loop abstraction)
examples/xor.py          — XOR training demonstration
```

---

## Why XOR Is the Right First Test

XOR is not linearly separable:

```
(0,0)=0    (0,1)=1
(1,0)=1    (1,1)=0
```

Plot these four points. No single straight line separates the 1s from the 0s. A perceptron (single neuron, linear) provably cannot solve it — this is the Minsky-Papert result from 1969, which temporarily killed neural network research.

A two-layer MLP with nonlinear activation **can** solve it, because the hidden layer learns to transform the input space into one that is linearly separable. If Sonnet's XOR accuracy reaches 100%, every component in the stack is proven correct:

- Tensor arithmetic ✓
- Computational graph construction ✓
- Topological sort ✓
- Chain rule / backward propagation ✓
- Neuron, Layer, MLP ✓
- Loss function ✓
- Gradient descent ✓

---

## Loss Functions

### MSELoss

```
L = (1/n) * Σ (pred_i - target_i)²
```

Implementation detail: we build the loss as a `Tensor` expression using `+`, `-`, `**`. This means the autograd graph extends through the loss. When `loss.backward()` runs, gradients flow back through the loss and into every model parameter automatically.

```python
diff = pred - tgt       # Tensor subtraction
total = total + diff**2 # Tensor pow and add
loss = total * Tensor(1.0 / n)
```

Every operation creates a node. The loss is just another node in the same graph that already contains the entire network.

### BinaryCrossEntropy

```
L = -(1/n) * Σ [ y*log(p) + (1-y)*log(1-p) ]
```

Why BCE beats MSE for classification: compare the loss on a confident wrong prediction (pred=0.99, target=0):

| Loss | Value |
|------|-------|
| MSE  | (0.99 - 0)² = 0.98 |
| BCE  | -log(0.01) ≈ 4.6   |

BCE punishes confident wrong answers ~5× harder. This creates much stronger gradient signal and faster correction.

**Numerical stability:** `log(0) = -∞`. We clamp predictions to `[ε, 1-ε]` where `ε = 1e-7`. Standard in every framework.

---

## The Training Loop

The Trainer executes the same five-step cycle every framework uses:

```
for each step:
    1. predictions = model(X)          # forward pass
    2. loss = loss_fn(predictions, y)  # compute scalar loss
    3. model.zero_grad()               # clear old gradients
    4. loss.backward()                 # compute new gradients
    5. p.data -= lr * p.grad           # gradient descent
```

### Why step order matters

`zero_grad()` must come before `backward()`. If you call `backward()` twice without clearing gradients, they accumulate from both calls — the parameter update will be wrong.

`backward()` must come before the parameter update — gradients must exist before you use them.

### Why `p.data -= lr * p.grad` not `p -= lr * p.grad`

`p -= ...` would create a new `Tensor` object and reassign the variable. The old `Tensor` — the one referenced inside the computational graph — would be unaffected. The model would never actually update.

`p.data -= ...` mutates the scalar inside the existing `Tensor` object in-place. The object reference is preserved. Next forward pass, the same `Tensor` objects in the graph carry the updated values.

---

## Learning Rate Sensitivity

XOR with `MLP(2, [4, 1])` and `MSELoss`:

| Learning rate | Behavior |
|---------------|----------|
| 0.001         | Very slow, needs 2000+ steps |
| 0.05          | Reliable convergence ~300 steps |
| 0.1           | Fast, converges ~150-300 steps |
| 0.5           | Often overshoots, loss oscillates |
| 1.0           | Usually diverges |

`lr=0.1` with `steps=500` is the reliable setting used here.

---

## What the Decision Boundary Probe Shows

```python
for xv in [0.0, 0.25, 0.5, 0.75, 1.0]:
    for yv in [0.0, 0.25, 0.5, 0.75, 1.0]:
        out = model([xv, yv])
```

A linear model would produce a straight boundary. Sonnet's trained MLP produces a **curved, non-linear boundary** — the signature of a model that actually learned the XOR structure.

```
░ █ █ █ ░       (0,0)=0 and (1,1)=0 shown as ░
█ █ ░ █ █       (0,1)=1 and (1,0)=1 shown as █
█ ░ ░ ░ █
█ █ ░ █ █
░ █ █ █ ░
```

The curved checkerboard pattern is the visual proof.

---

## Architecture Summary

```
sonnet/
├── sonnet_core/
│   ├── tensor.py     — Tensor + autograd        (stage 1)
│   ├── nn.py         — Neuron, Layer, MLP        (stage 2)
│   ├── loss.py       — MSELoss, BCE              (stage 3)  ← new
│   └── trainer.py    — Trainer                   (stage 3)  ← new
├── examples/
│   └── xor.py        — XOR demo                  (stage 3)  ← new
└── tests/
    ├── test_backward.py
    └── test_nn.py
```

---

## Run

```bash
cd sonnet
python -m examples.xor
```

---

## Next Step

With XOR solved, the foundation is proven. The next architectural decision is:

**N-dimensional Tensors** — replacing scalar `data: float` with `data: list[list[...]]` backed by a proper ndarray implementation.

This is the most significant upgrade in the framework's history. It unlocks:
- Batch training (multiple samples per step, faster and more stable)
- Matrix multiplication (the core of every layer)
- Convolutional operations (images)
- Attention (transformers)

Every subsequent component — optimizers, batch norm, embeddings, transformers — requires this foundation.

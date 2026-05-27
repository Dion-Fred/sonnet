# Sonnet — Stage 5: Optimizers

## What Was Built

```
sonnet_core/optim.py     — SGD (with momentum + weight decay), Adam
tests/test_optim.py      — 10 tests
```

---

## The Problem With Raw Gradient Descent

Every training step so far used:

```python
p.data[i][j] -= lr * p.grad[i][j]
```

This works. But it has three fundamental problems:

**1. Oscillation in ravine landscapes**

Loss surfaces are rarely spherical. They're elongated ravines — steep in one direction, shallow in another. Vanilla SGD overshoots across the ravine while crawling along it. Each step wastes most of its movement on oscillation.

**2. Same learning rate for every parameter**

A weight connected to a frequent feature needs small, careful updates. A weight connected to a rare feature needs large updates to learn anything. One global `lr` is a compromise that's wrong for both.

**3. Manual tuning sensitivity**

`lr=0.1` diverges. `lr=0.001` never converges. The right value shifts as training progresses. Vanilla SGD gives you no help.

Momentum solves problem 1. Adam solves all three.

---

## SGD with Momentum

```
v_t = β * v_{t-1} + (1 - β) * grad_t
p_t = p_{t-1} - lr * v_t
```

`v` is velocity — an exponentially weighted moving average of all past gradients.

`β = 0.9` means: 90% of last step's direction + 10% new gradient.

**Why this kills oscillation:**

In the steep direction (across the ravine), gradients alternate sign (+, -, +, -). The velocity averages them: `0.1 + 0.1*(-1) + ... → 0`. Oscillation damps out.

In the shallow direction (along the ravine), gradients consistently point the same way. Velocity accumulates: `0.1 + 0.1 + 0.1 + ... → builds up`. Progress accelerates.

The result: movement is redirected from oscillation into the true descent direction.

---

## Adam

```
m_t = β1*m + (1-β1)*grad          # first moment: where are we going?
v_t = β2*v + (1-β2)*grad²         # second moment: how noisy is this direction?

m̂ = m / (1 - β1^t)               # bias-corrected mean
v̂ = v / (1 - β2^t)               # bias-corrected variance

p = p - lr * m̂ / (sqrt(v̂) + ε)
```

`m` is momentum — smoothed gradient direction.
`v` is smoothed squared gradient — a measure of how volatile each parameter's gradient has been.

**Dividing by `sqrt(v̂)` is the key insight:**

A parameter whose gradient fluctuates wildly gets a small step (large `v`, small update). A parameter whose gradient is consistently small and stable gets a larger step (small `v`, large update). Every parameter has its own effective learning rate, adapted from history.

**Bias correction:**

`m` and `v` start at zero. At step 1:
- `m = (1-0.9) * grad = 0.1 * grad` — ten times too small
- Without correction, the first update would be tiny and distorted

Dividing by `(1 - β^t)`:
- Step 1: divide by `0.1` — restores true scale
- Step 100: divide by `≈1.0` — no effect (correction fades away)

**Default hyperparameters work for almost everything:**
- `β1 = 0.9`
- `β2 = 0.999`
- `ε  = 1e-8`
- `lr = 1e-3`

The only thing you typically tune is `lr`.

---

## In-place Mutation Pattern

Optimizer state (`m`, `v`, `velocity`) and parameter data are mutated in-place using the `_apply_inplace` / `_map2_inplace` / `_map3_inplace` utilities. 

Why in-place? The same reason as `p.data -= lr * p.grad` from Stage 3: the `NDTensor` object must remain the same Python object. If we created a new object, the model's internal references would point to the old (unupdated) tensor.

The utilities recurse through nested lists, applying scalar functions at the leaves. The pattern:

```python
_map2_inplace(target, a, b, fn)
# target[i][j] = fn(a[i][j], b[i][j]) for all i, j
```

---

## Weight Decay

Both optimizers support `weight_decay` (L2 regularization):

```python
effective_grad = grad + λ * param
```

Adding `λ * param` to the gradient is equivalent to adding `(λ/2) * ||param||²` to the loss. This penalizes large weights, which:
- Reduces overfitting (model can't rely on any single large weight)
- Keeps weights small, which helps gradient flow in deep networks
- Equivalent to "weight decay" — each step slightly shrinks all weights toward zero

Typical value: `1e-4` to `1e-2`.

---

## Standard Training Loop (Final Form)

```python
model     = Sequential(Linear(2, 4), Tanh(), Linear(4, 1))
loss_fn   = MSELoss()
optimizer = Adam(model.parameters(), lr=1e-3)

for step in range(steps):
    pred = model(x)
    loss = loss_fn(pred, target)

    optimizer.zero_grad()   # clear gradients
    loss.backward()         # compute gradients
    optimizer.step()        # update parameters

    if step % 10 == 0:
        print(f"step {step}: loss={loss.data:.4f}")
```

This is the canonical form. Every serious deep learning project uses exactly this loop.

---

## Full Architecture So Far

```
sonnet/
├── sonnet_core/
│   ├── tensor.py      — scalar autograd            (stage 1)
│   ├── nn.py          — scalar MLP                 (stage 2)
│   ├── loss.py        — scalar loss functions      (stage 3)
│   ├── trainer.py     — training loop              (stage 3)
│   ├── ndtensor.py    — matrix autograd            (stage 4)
│   ├── linear.py      — Linear, Sequential         (stage 4)
│   └── optim.py       — SGD, Adam                  (stage 5) ← new
├── examples/
│   └── xor.py
└── tests/
    ├── test_backward.py
    ├── test_nn.py
    ├── test_ndtensor.py
    └── test_optim.py                               ← new
```

---

## Run Tests

```bash
cd sonnet
python -m tests.test_optim
```

---

## Next Step: Stage 6 — Data Pipeline + Real Training

With optimizers complete, the training infrastructure is finished. What's missing is the ability to train on real datasets with proper batching.

**Stage 6 builds:**

```
sonnet_core/data.py    — Dataset, DataLoader (batching + shuffling)
examples/mnist.py      — train on flattened MNIST (784 → 10 classes)
```

**Why batching matters:**

Single-sample updates are noisy — one sample's gradient may point in the wrong direction. Averaging gradients over a batch of 32 or 64 samples gives a better estimate of the true gradient direction, and makes better use of the computation done per step.

After Stage 6, Sonnet trains on real data for the first time. After that: the transformer.

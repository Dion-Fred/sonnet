# Sonnet — Stage 4: NDTensor and Linear Layer

## What Was Built

```
sonnet_core/ndtensor.py   — N-dimensional tensor with full autograd
sonnet_core/linear.py     — Linear layer, Sequential, activations, MSELoss
tests/test_ndtensor.py    — 20 tests
```

---

## Why This Stage Is the Hardest

The scalar `Tensor` had one value, one gradient. Every operation was trivially differentiable.

`NDTensor` has grids of values. Operations like matrix multiply mix values across rows and columns. The gradient of a matrix operation is itself a matrix operation — and getting the shapes right requires thinking carefully about what "upstream gradient" means when it's a matrix, not a scalar.

---

## Data Representation

```python
# Scalar
Tensor(3.0)           # data = 3.0

# 1D vector (bias)
NDTensor([1.0, 2.0])  # data = [1.0, 2.0],  shape = (2,)

# 2D matrix (weight, batch)
NDTensor([[1.0, 2.0],
          [3.0, 4.0]]) # shape = (2, 2)
```

Pure nested Python lists. No numpy. Shape is computed recursively at construction time.

---

## The Critical Operation: Matrix Multiply

```
Z = A @ B
A: (m, k)
B: (k, n)
Z: (m, n)
```

Each element: `Z[i][j] = sum over p of A[i][p] * B[p][j]`

### Gradients via chain rule

Given upstream gradient `dL/dZ` (same shape as Z, `m×n`):

```
dL/dA = dL/dZ @ B.T      shape: (m,n) @ (n,k) = (m,k) ✓ matches A
dL/dB = A.T @ dL/dZ      shape: (k,m) @ (m,n) = (k,n) ✓ matches B
```

These are not arbitrary formulas. They fall directly out of the chain rule applied to the sum:

```
∂Z[i][j]/∂A[i][p] = B[p][j]

dL/dA[i][p] = sum_j( dL/dZ[i][j] * B[p][j] )
            = (dL/dZ @ B.T)[i][p]
```

Matrix multiply is its own gradient operation. This is why `@` is the foundation of all neural network math.

---

## Broadcast Add (Bias)

```python
A: shape (m, n)   — batch of activations
b: shape (n,)     — bias vector

C = A + b         — each row of A gets b added
```

Forward: straightforward row-wise addition.

Backward: upstream gradient has shape `(m, n)`. Bias only has `n` values. We must **sum over rows** to get the bias gradient:

```python
b.grad += sum_axis0(out.grad)   # shape (n,)
```

Why sum? Because each element `b[j]` contributed to every row. Chain rule: sum all those contributions.

---

## Kaiming Initialization

```python
k = sqrt(1 / in_features)
W ~ Uniform(-k, k)
```

Why does initialization matter so much?

- Too large: activations saturate (output ≈ ±1 for tanh), gradients ≈ 0, learning stops
- Too small: signal shrinks through each layer, gradients vanish before reaching early layers
- Kaiming: calibrated so variance is roughly preserved through a relu layer

For a layer with `in_features=100`: `k = 0.1`, weights in `[-0.1, 0.1]`. Small but not too small.

---

## Box-Muller Normal Sampling

```python
z = sqrt(-2 * log(u1)) * cos(2π * u2)
```

Transforms two uniform random numbers into one standard normal sample. Used in `NDTensor.randn()`. No numpy, no `random.gauss` — pure math from first principles. This matters because weight initialization quality affects training stability.

---

## Linear Layer

```
output = input @ W.T + b

input : (batch, in_features)
W     : (out_features, in_features)
W.T   : (in_features, out_features)
output: (batch, out_features)
```

W stores one neuron per row (each row = one set of weights). Transposing before multiply produces the right output shape. This is the PyTorch `nn.Linear` convention exactly.

Single-sample input `(in_features,)` is auto-promoted to `(1, in_features)`.

---

## Sequential

```python
model = Sequential(
    Linear(2, 4),
    Tanh(),
    Linear(4, 1),
)
out = model(x)   # passes x through each layer in order
```

Output of each layer is input to the next. Parameters are collected from all layers. The entire model is one unified computational graph — `loss.backward()` traces through every layer automatically.

---

## What Changed in the Autograd Engine

Nothing. The topological sort, backward walk, and `_backward` closure pattern are identical to the scalar `Tensor`. Only the data and gradient math changed.

This is the payoff of the modular design from Stage 1: the autograd engine is completely decoupled from the shape of data it operates on.

---

## File Structure

```
sonnet/
├── sonnet_core/
│   ├── tensor.py      — scalar autograd      (stage 1)
│   ├── nn.py          — scalar MLP           (stage 2)
│   ├── loss.py        — scalar loss          (stage 3)
│   ├── trainer.py     — training loop        (stage 3)
│   ├── ndtensor.py    — matrix autograd      (stage 4) ← new
│   └── linear.py      — Linear, Sequential   (stage 4) ← new
├── examples/
│   └── xor.py
└── tests/
    ├── test_backward.py
    ├── test_nn.py
    └── test_ndtensor.py                                 ← new
```

---

## Run Tests

```bash
cd sonnet
python -m tests.test_ndtensor
```

---

## Next Step: Stage 5 — Optimizers

With NDTensor working, the parameter update loop:

```python
for p in model.parameters():
    p.data[i][j] -= lr * p.grad[i][j]
```

Needs to become a proper `Optimizer` object. Two to build:

**SGD with momentum:**
```
velocity = momentum * velocity - lr * grad
param   += velocity
```
Momentum accumulates gradient direction over steps — faster convergence, escapes shallow local minima.

**Adam:**
```
m = β1*m + (1-β1)*grad          # first moment (mean)
v = β2*v + (1-β2)*grad²         # second moment (variance)
param -= lr * m / (sqrt(v) + ε) # normalized update
```
Adam is what trains almost every modern model. Adaptive learning rates per parameter. Robust to learning rate choice.

After optimizers: a real dataset (MNIST digits, flattened to 784 features) — the first test on data that wasn't hand-crafted.

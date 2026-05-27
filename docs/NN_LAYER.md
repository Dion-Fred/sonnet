# Sonnet — Neural Network Layer (nn.py)

## What Was Built

`sonnet_core/nn.py` contains three composable components:

```
Neuron  →  Layer  →  MLP
```

All built entirely on `Tensor`. No new autograd code was needed — the engine from the previous step handles everything automatically.

---

## Component Hierarchy

```
MLP
 └── Layer[]
      └── Neuron[]
           ├── w: Tensor[]   (one per input)
           └── b: Tensor     (scalar bias)
```

---

## Neuron

```python
n = Neuron(nin=3, activation='tanh')
output = n([x1, x2, x3])
```

**What it computes:**

```
output = tanh( w0*x0 + w1*x1 + w2*x2 + b )
```

**Parameter count:** `nin + 1` (weights + bias)

**Weight initialization:** `uniform(-1, 1)`

Why random? The symmetry problem. If all weights start equal, all neurons in a layer compute the same output, receive the same gradient, and update identically — they never differentiate. Randomness breaks this symmetry from step one.

Why small? Large initial weights saturate `tanh`, pushing outputs to ±1 where the gradient `1 - tanh²(x)` approaches 0. Training stalls. Small weights keep neurons in the active, high-gradient region of the activation function.

---

## Activation Functions

### tanh

```
tanh(x) = (e^x - e^-x) / (e^x + e^-x)

Range:    (-1, 1)
d/dx:     1 - tanh(x)²
```

The gradient `1 - tanh(x)²` is:
- Maximum (= 1) at `x = 0`
- Approaches 0 as `|x| → ∞` (saturation)

This is why weight initialization matters — we want activations near zero at initialization.

### relu

```
relu(x) = max(0, x)

Range:    [0, ∞)
d/dx:     1 if x > 0, else 0
```

Simpler gradient. No saturation on the positive side. Can produce "dead neurons" (permanently zero if weights push input negative and never recover). Useful to have both available.

---

## Module Base Class

All components inherit from `Module`:

```python
class Module:
    def parameters(self): return []
    def zero_grad(self): ...
    def __call__(self, *args): return self.forward(*args)
```

`__call__` delegates to `forward()`. This means `model(x)` and `model.forward(x)` are equivalent — same convention as PyTorch.

`zero_grad()` resets all parameter gradients before each backward pass. Without this, gradients accumulate across training steps (which is sometimes intentional, but not usually).

---

## Layer

```python
l = Layer(nin=3, nout=4)
outputs = l([x0, x1, x2])  # returns list of 4 Tensors
```

A layer is just `nout` neurons sharing the same input vector, each producing one scalar output.

**Parameter count:** `nout * (nin + 1)`

For `Layer(3, 4)`: `4 * (3 + 1) = 16` parameters.

---

## MLP

```python
model = MLP(nin=2, nouts=[4, 4, 1])
output = model([x0, x1])  # single Tensor
```

**Architecture:**

```
Input (2)
  → Layer(2→4, tanh)
  → Layer(4→4, tanh)
  → Layer(4→1, linear)
Output (1 scalar)
```

The last layer uses `activation=None` (linear) by convention. For regression this is correct — we don't want to clip the output to (-1, 1). For classification, apply a sigmoid or tanh externally on the output.

**Parameter count for MLP(2, [4, 4, 1]):**
- Layer 0: `4 * (2+1) = 12`
- Layer 1: `4 * (4+1) = 20`
- Layer 2: `1 * (4+1) = 5`
- **Total: 37 parameters**

---

## Training Loop Pattern

This is the complete, correct pattern for one training step:

```python
model = MLP(2, [4, 1])

for step in range(100):
    # 1. Forward pass
    pred = model(x)

    # 2. Compute loss
    loss = (pred - target) ** 2

    # 3. Zero gradients (must happen before backward)
    model.zero_grad()

    # 4. Backward pass — computes all gradients
    loss.backward()

    # 5. Gradient descent step
    for p in model.parameters():
        p.data -= learning_rate * p.grad
```

**Why zero_grad before backward, not after?**

Personal preference — either position works as long as it happens before the next `backward()`. Before is more natural: "clean slate, then compute."

**Why `p.data -= lr * p.grad` and not `p -= lr * p.grad`?**

`p -= ...` would create a new Tensor and break the reference. `p.data -= ...` modifies the existing Tensor in-place, so `model.parameters()` still points to the same objects next step.

---

## What the Gradient Descent Step Does

```
p.data -= lr * p.grad
```

`p.grad` holds `∂loss/∂p` — how much the loss increases per unit increase in `p`.

Subtracting it (scaled by learning rate) moves `p` in the direction that reduces loss. That's gradient descent.

`lr` (learning rate) controls step size:
- Too large → overshoots, loss diverges
- Too small → training is slow
- Typical starting values: 0.01 – 0.1

---

## Files Added

```
sonnet/
├── sonnet_core/
│   └── nn.py            ← Neuron, Layer, MLP, tanh, relu
└── tests/
    └── test_nn.py       ← 12 tests
```

---

## Next Step

With a working MLP, the next component is:

**A real training example** — binary classification on XOR or a simple dataset.

XOR is the classic first test because:
- It is not linearly separable (a single neuron cannot solve it)
- A two-layer MLP with nonlinearity can solve it perfectly
- It proves that the entire stack — forward, backward, parameter update — is actually working

After that: **loss functions** (`MSE`, `binary cross-entropy`) as proper `Module` objects, then a **training loop abstraction** (`Trainer`).

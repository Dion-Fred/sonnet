# Sonnet — Backward Propagation System

## What Was Built

`sonnet_core/tensor.py` now contains a complete scalar autograd engine:

- Forward operations: `+`, `-`, `*`, `/`, `**`, unary negation
- Scalar interop: Python `int`/`float` on either side of any operator
- `backward()` — full reverse-mode automatic differentiation
- `zero_grad()` — gradient reset utility

---

## Core Concept: The `_backward` Closure

Every operation creates an output tensor **and** registers a `_backward` function on it.

```python
def __mul__(self, other):
    out = Tensor(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad  += other.data * out.grad
        other.grad += self.data  * out.grad

    out._backward = _backward
    return out
```

Key details:

1. **`+=` not `=`** — gradients accumulate. If a tensor is used multiple times in a graph, each usage path contributes. Using `=` would destroy earlier contributions.
2. **Closure captures `out`, `self`, `other`** — Python closures capture by reference, so `out.grad` inside `_backward` always reads the gradient that has been written to `out` by the time `_backward` is called.
3. **Local gradient × upstream gradient** — this is the chain rule. `other.data` is the local derivative; `out.grad` is the upstream gradient flowing back.

---

## Gradient Reference Table

| Operation | Expression | `∂out/∂self` | `∂out/∂other` |
|-----------|------------|--------------|----------------|
| Add       | `x + y`    | `1`          | `1`            |
| Sub       | `x - y`    | `1`          | `-1`           |
| Mul       | `x * y`    | `y`          | `x`            |
| Div       | `x / y`    | `1/y`        | `-x/y²`        |
| Pow       | `x**n`     | `n·xⁿ⁻¹`    | —              |
| Neg       | `-x`       | `-1`         | —              |

---

## Topological Sort

```python
def build_topo(tensor):
    if tensor not in visited:
        visited.add(tensor)
        for parent in tensor._prev:
            build_topo(parent)
        topo.append(tensor)   # append AFTER processing all parents
```

Why this ordering matters:

- A tensor is appended **after** all tensors it depends on
- So `topo` is ordered inputs → output (leaves first)
- `reversed(topo)` gives output → inputs
- This guarantees every tensor's gradient is fully accumulated **before** we call its `_backward()`

If we skipped this and processed in arbitrary order, `out.grad` might still be `0.0` when `_backward()` runs — giving wrong results.

---

## Gradient Seeding

```python
self.grad = 1.0
```

This represents `dL/dL = 1`. The output tensor is the "loss" — its gradient with respect to itself is always 1. Every other gradient flows from this seed.

---

## Gradient Accumulation and Tensor Reuse

```python
x = Tensor(3.0)
z = x * x
z.backward()
# x.grad == 6.0
```

When `x` appears as both `self` and `other` in the multiply, `_backward()` runs:

```python
self.grad  += other.data * out.grad   # += 3.0
other.grad += self.data  * out.grad   # += 3.0
```

Both lines target `x.grad`. The result is `6.0 = 2x`, which is exactly `d(x²)/dx`. The `+=` accumulation is not an accident — it is the mechanically correct thing to do.

---

## Chain Rule Example

```python
x = Tensor(2.0)
y = Tensor(3.0)
w = Tensor(4.0)

s = x + y    # s = 5
z = s * w    # z = 20

z.backward()
```

Graph:

```
x --\
     (+) -- s --\
y --/             (*) -- z
              w --/
```

Backward pass (reversed topo order: z, s, x, y, w):

1. `z.grad = 1.0`
2. `z._backward()`: `s.grad += w.data * 1.0 = 4.0`, `w.grad += s.data * 1.0 = 5.0`
3. `s._backward()`: `x.grad += 1.0 * 4.0 = 4.0`, `y.grad += 1.0 * 4.0 = 4.0`
4. Leaf tensors (`x`, `y`, `w`) have no-op `_backward`

Result:

| Tensor | Gradient | Meaning |
|--------|----------|---------|
| `x`    | `4.0`    | `dz/dx = w` |
| `y`    | `4.0`    | `dz/dy = w` |
| `w`    | `5.0`    | `dz/dw = x+y` |

---

## Scalar Interop

All operators handle Python scalars via `__r*__` methods:

```python
x = Tensor(5.0)
z = 3 * x    # calls x.__rmul__(3), which calls x.__mul__(Tensor(3))
```

This means Sonnet tensors compose naturally with Python numeric code.

---

## Architecture Notes

- **Scalar only (for now)** — `data` is a single `float`. N-dimensional tensor support comes later when we add `ndarray` backing.
- **Reverse-mode AD** — the correct mode for neural networks where scalar loss differentiates with respect to many parameters. Forward-mode would be inefficient.
- **No tape** — the graph is built implicitly through Python object references. There is no explicit "tape" object. This matches PyTorch's eager-mode design.

---

## Files Changed

```
sonnet/
├── sonnet_core/
│   └── tensor.py      ← full rewrite with backward()
└── tests/
    └── test_backward.py  ← 10 tests covering all operations and edge cases
```

---

## Next Step

With autograd working on scalars, the next component is:

**`sonnet_core/nn.py`** — a `Neuron`, `Layer`, and `MLP` class built on top of `Tensor`.

A `Neuron` is:

```
output = activation(sum(wi * xi) + b)
```

Where every `wi` and `b` is a `Tensor`, meaning the entire network is differentiable end-to-end through the autograd system built here.

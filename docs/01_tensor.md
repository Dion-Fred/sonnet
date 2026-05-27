# Sonnet Tensor System

## Overview

The Tensor class is the foundation of the Sonnet deep learning framework.

All future operations:
- matrix math
- gradients
- activations
- neural networks

will use Tensor objects.

---

# Current Implementation

```python
class Tensor:

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Tensor({self.data})"
```

---

# Features

Current tensor supports:
- storing numerical data
- readable printing

---

# Example

```python
x = Tensor(5)

print(x)
```

Output:

```python
Tensor(5)
```

---

# Why Tensors Matter

Deep learning frameworks operate on tensors.

Examples:
- PyTorch
- TensorFlow
- JAX

all use tensors as their core abstraction.

Sonnet begins with the same design principle.

---

# Current Limitations

Tensor currently:
- cannot perform arithmetic
- has no gradients
- has no computational graph

---

# Next Step

Implement tensor arithmetic:
- addition
- subtraction
- multiplication
- division

Target usage:

```python
x = Tensor(2)
y = Tensor(3)

z = x + y

print(z)
```

Expected output:

```python
Tensor(5)
```
# Sonnet Autograd Graph System

## Overview

Sonnet tensors now track:
- gradients
- parent tensors
- operations

This introduces computational graph construction.

Computational graphs are the foundation of:
- automatic differentiation
- backpropagation
- neural network training

---

# Updated Tensor Structure

```python
class Tensor:

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0

        self._prev = set(_children)

        self._op = _op
```

---

# New Tensor Properties

## grad

Stores gradient value.

Example:

```python
x.grad
```

Initially gradients are zero.

Later they will store:
- derivatives
- gradient flow information

---

## _prev

Stores tensors used to create current tensor.

Example:

```python
z = x + y
```

Then:

```python
z._prev == {x, y}
```

This creates graph connectivity.

---

## _op

Stores operation that created tensor.

Example:

```python
z = x * y
```

Then:

```python
z._op == '*'
```

This helps determine:
- backward propagation rules
- gradient behavior

---

# Example

```python
x = Tensor(2)
y = Tensor(3)

z = x * y
```

Graph:

```text
x ----\
       *
y ----/ \
         z
```

---

# Why Computational Graphs Matter

Neural networks require gradients.

Gradients are computed by:
- tracing operations backward
- applying chain rule
- propagating derivatives

Computational graphs store the information required for this process.

---

# Current Limitations

Current Sonnet:
- tracks graph structure
- stores operations
- stores gradients

But:
- gradients are not computed yet
- backward propagation does not exist
- chain rule is not implemented

---

# Next Step

Implement:
- backward propagation
- chain rule
- gradient computation

Goal:

```python
x = Tensor(2)
y = Tensor(3)

z = x * y

z.backward()

print(x.grad)
print(y.grad)
```

Expected behavior:

```text
dz/dx = y
dz/dy = x
```

Meaning:

```text
x.grad = 3
y.grad = 2
```

This will become Sonnet’s first true learning mechanism.
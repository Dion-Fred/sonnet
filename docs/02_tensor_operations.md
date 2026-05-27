# Sonnet Tensor Arithmetic

## Overview

Sonnet tensors now support arithmetic operations.

Implemented operations:
- addition
- subtraction
- multiplication
- division

This allows tensors to participate in mathematical computation.

---

# Current Implementation

```python
class Tensor:

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Tensor({self.data})"

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __sub__(self, other):
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        return Tensor(self.data * other.data)

    def __truediv__(self, other):
        return Tensor(self.data / other.data)
```

---

# Operator Overloading

Python allows objects to define behavior for operators.

Example:

```python
x + y
```

calls:

```python
x.__add__(y)
```

internally.

Sonnet uses this mechanism to create tensor arithmetic.

---

# Example

```python
x = Tensor(10)
y = Tensor(5)

print(x + y)
print(x - y)
print(x * y)
print(x / y)
```

Output:

```python
Tensor(15)
Tensor(5)
Tensor(50)
Tensor(2.0)
```

---

# Why This Matters

Deep learning frameworks rely heavily on tensor operations.

Neural networks are essentially:
- tensor transformations
- matrix computations
- gradient propagation

Tensor arithmetic is the first step toward:
- computational graphs
- automatic differentiation
- backpropagation

---

# Current Limitations

Current tensor:
- only supports scalar values
- has no gradients
- cannot handle matrices or vectors
- has no graph tracking

---

# Next Step

Implement:
- gradient tracking
- computational graph storage

Goal:

```python
x = Tensor(2)
y = Tensor(3)

z = x * y

print(z.grad)
```

Future tensors will understand:
- how they were created
- which operation produced them
- how gradients flow backward
"""
sonnet/sonnet_core/ndtensor.py

Sonnet NDTensor — N-dimensional Tensor with Autograd

Replaces the scalar Tensor for all serious computation.
data is stored as a nested Python list (no numpy).
Shapes are tracked explicitly.

Supported operations (all differentiable):
    add, sub, mul (element-wise)
    matmul         (@)
    transpose      (.T)
    sum            (reduce to scalar or along axis)
    mean
    relu, tanh     (element-wise activations)
    __pow__        (element-wise power)
    __neg__

Design philosophy:
    Every operation records a _backward closure — identical
    to the scalar Tensor. The autograd engine (topological sort,
    backward walk) is unchanged. Only the data representation
    and gradient math change.
"""

import math


# ------------------------------------------------------------------
# Low-level list math utilities
# No numpy. Pure Python.
# ------------------------------------------------------------------

def _shape(data):
    """Recursively determine shape of nested list."""
    if not isinstance(data, list):
        return ()
    if len(data) == 0:
        return (0,)
    return (len(data),) + _shape(data[0])


def _zeros(shape):
    """Create a nested list of zeros with given shape."""
    if len(shape) == 0:
        return 0.0
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [_zeros(shape[1:]) for _ in range(shape[0])]


def _ones(shape):
    """Create a nested list of ones with given shape."""
    if len(shape) == 0:
        return 1.0
    if len(shape) == 1:
        return [1.0] * shape[0]
    return [_ones(shape[1:]) for _ in range(shape[0])]


def _copy(data):
    """Deep copy a nested list."""
    if not isinstance(data, list):
        return float(data)
    return [_copy(d) for d in data]


def _map2(fn, a, b):
    """Element-wise binary operation on two nested lists of same shape."""
    if not isinstance(a, list):
        return fn(a, b)
    return [_map2(fn, ai, bi) for ai, bi in zip(a, b)]


def _map1(fn, a):
    """Element-wise unary operation on a nested list."""
    if not isinstance(a, list):
        return fn(a)
    return [_map1(fn, ai) for ai in a]


def _add_inplace(a, b):
    """a += b element-wise, mutates a."""
    if not isinstance(a, list):
        # Can't mutate a float in-place at base level;
        # caller handles scalar case
        return a + b
    for i in range(len(a)):
        if isinstance(a[i], list):
            _add_inplace(a[i], b[i])
        else:
            a[i] += b[i]
    return a


def _matmul(A, B):
    """
    2D matrix multiply: A (m×k) @ B (k×n) → C (m×n)

    C[i][j] = sum over k of A[i][k] * B[k][j]
    """
    m = len(A)
    k = len(A[0])
    n = len(B[0])
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            C[i][j] = s
    return C


def _transpose(A):
    """Transpose a 2D list: (m×n) → (n×m)"""
    m = len(A)
    n = len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def _sum_all(data):
    """Sum all elements in a nested list → float."""
    if not isinstance(data, list):
        return float(data)
    return sum(_sum_all(d) for d in data)


def _sum_axis0(A):
    """
    Sum 2D matrix along axis 0 (over rows) → 1D list of length n.
    Shape (m, n) → (n,)
    Used for bias gradients.
    """
    m = len(A)
    n = len(A[0])
    result = [0.0] * n
    for i in range(m):
        for j in range(n):
            result[j] += A[i][j]
    return result


def _broadcast_add(A, b):
    """
    Add bias vector b (shape n,) to each row of A (shape m×n).
    Returns new list of shape (m, n).
    """
    return [[A[i][j] + b[j] for j in range(len(b))] for i in range(len(A))]


def _scalar_mul(data, scalar):
    """Multiply all elements of nested list by scalar."""
    return _map1(lambda x: x * scalar, data)


def _outer_broadcast_grad(grad, original_shape, target_shape):
    """
    When a smaller tensor was broadcast to a larger shape,
    sum the gradient back down to the original shape.
    """
    # If original was 1D (bias), sum over axis 0
    if len(original_shape) == 1 and len(target_shape) == 2:
        return _sum_axis0(grad)
    return grad


# ------------------------------------------------------------------
# NDTensor
# ------------------------------------------------------------------

class NDTensor:
    """
    N-dimensional tensor with reverse-mode automatic differentiation.

    data  : nested Python list (e.g. [[1,2],[3,4]] for 2×2 matrix)
    shape : tuple (e.g. (2, 2))
    grad  : same shape as data, initialized to zeros

    Usage:
        A = NDTensor([[1.0, 2.0], [3.0, 4.0]])
        B = NDTensor([[1.0], [1.0]])
        C = A @ B
        C_sum = C.sum()
        C_sum.backward()
        # A.grad and B.grad now contain correct gradients
    """

    def __init__(self, data, _children=(), _op='', _label=''):
        # Normalize: convert ints/floats to float throughout
        if isinstance(data, (int, float)):
            data = float(data)
        elif isinstance(data, list):
            data = _map1(float, data)

        self.data  = data
        self.shape = _shape(data) if isinstance(data, list) else ()
        self.grad  = _zeros(self.shape) if self.shape else 0.0

        self._backward = lambda: None
        self._prev     = set(_children)
        self._op       = _op
        self._label    = _label

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return f"NDTensor(shape={self.shape}, op='{self._op}')"

    def item(self):
        """Return scalar value. Only valid for shape () or (1,) or [[x]]."""
        if isinstance(self.data, float):
            return self.data
        if self.shape == (1,):
            return self.data[0]
        if self.shape == (1, 1):
            return self.data[0][0]
        raise ValueError(f"item() called on non-scalar NDTensor of shape {self.shape}")

    # ------------------------------------------------------------------
    # Element-wise Operations
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, NDTensor) else NDTensor(other)

        # Handle broadcast: (m,n) + (n,) adds bias to each row
        if self.shape != other.shape:
            if (len(self.shape) == 2 and len(other.shape) == 1
                    and self.shape[1] == other.shape[0]):
                out_data = _broadcast_add(self.data, other.data)
                out = NDTensor(out_data, (self, other), '+broadcast')

                def _backward():
                    # gradient flows to self unchanged
                    self.grad = _map2(lambda g, s: g + s,
                                      out.grad, self.grad)
                    # gradient to bias: sum over rows
                    bias_grad = _sum_axis0(out.grad)
                    other.grad = _map2(lambda g, s: g + s,
                                       bias_grad, other.grad)

                out._backward = _backward
                return out
            else:
                raise ValueError(
                    f"Cannot add shapes {self.shape} and {other.shape}")

        out = NDTensor(_map2(lambda a, b: a + b, self.data, other.data),
                       (self, other), '+')

        def _backward():
            self.grad  = _map2(lambda g, s: g + s, out.grad, self.grad)
            other.grad = _map2(lambda g, s: g + s, out.grad, other.grad)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, NDTensor) else NDTensor(other)
        out = NDTensor(_map2(lambda a, b: a - b, self.data, other.data),
                       (self, other), '-')

        def _backward():
            self.grad  = _map2(lambda g, s: g + s,  out.grad, self.grad)
            other.grad = _map2(lambda g, s: g - s,  other.grad, out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        """Element-wise multiply. Also handles scalar * NDTensor."""
        other = other if isinstance(other, NDTensor) else NDTensor(
            _map1(lambda x: float(other) if not isinstance(other, NDTensor)
                  else x, self.data) if not isinstance(other, NDTensor)
            else other.data)

        # Scalar case
        if not isinstance(other.data, list):
            scalar = other.data
            out = NDTensor(_map1(lambda x: x * scalar, self.data),
                           (self, other), '*scalar')

            def _backward():
                self.grad = _map2(lambda g, s: g + s,
                                  _map1(lambda x: x * scalar, out.grad),
                                  self.grad)

            out._backward = _backward
            return out

        out = NDTensor(_map2(lambda a, b: a * b, self.data, other.data),
                       (self, other), '*')

        def _backward():
            self.grad  = _map2(lambda g, s: g + s,
                               _map2(lambda g, b: g * b, out.grad, other.data),
                               self.grad)
            other.grad = _map2(lambda g, s: g + s,
                               _map2(lambda g, a: g * a, out.grad, self.data),
                               other.grad)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        out = NDTensor(_map1(lambda x: -x, self.data), (self,), 'neg')

        def _backward():
            self.grad = _map2(lambda g, s: g - s, self.grad, out.grad)

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = NDTensor(
            _map1(lambda x: x ** exponent, self.data),
            (self,), f'**{exponent}')

        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            self.grad = _map2(
                lambda g, s: g + s,
                _map2(lambda g, x: g * exponent * (x ** (exponent - 1)),
                      out.grad, self.data),
                self.grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Matrix Multiply
    # ------------------------------------------------------------------

    def __matmul__(self, other):
        """
        Z = self @ other

        self  : (m, k)
        other : (k, n)
        Z     : (m, n)

        Gradients:
            dL/d(self)  = dL/dZ @ other.T
            dL/d(other) = self.T @ dL/dZ
        """
        assert len(self.shape) == 2 and len(other.shape) == 2, (
            f"matmul requires 2D tensors, got {self.shape} and {other.shape}")
        assert self.shape[1] == other.shape[0], (
            f"matmul shape mismatch: {self.shape} @ {other.shape}")

        out = NDTensor(_matmul(self.data, other.data), (self, other), '@')

        def _backward():
            # dL/dA = dL/dZ @ B^T
            dA = _matmul(out.grad, _transpose(other.data))
            self.grad = _map2(lambda g, s: g + s, dA, self.grad)

            # dL/dB = A^T @ dL/dZ
            dB = _matmul(_transpose(self.data), out.grad)
            other.grad = _map2(lambda g, s: g + s, dB, other.grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Transpose
    # ------------------------------------------------------------------

    @property
    def T(self):
        """Transpose: (m, n) → (n, m)"""
        assert len(self.shape) == 2, "T only supported for 2D tensors"
        out = NDTensor(_transpose(self.data), (self,), 'T')

        def _backward():
            # gradient of transpose is transpose of gradient
            self.grad = _map2(lambda g, s: g + s,
                              _transpose(out.grad), self.grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def sum(self):
        """Sum all elements → scalar NDTensor."""
        total = _sum_all(self.data)
        out = NDTensor(total, (self,), 'sum')

        def _backward():
            # gradient distributes uniformly to all elements
            self.grad = _map2(lambda g, s: g + s,
                              _map1(lambda _: out.grad, self.data),
                              self.grad)

        out._backward = _backward
        return out

    def mean(self):
        """Mean of all elements → scalar NDTensor."""
        n = _sum_all(_map1(lambda _: 1.0, self.data))
        total = _sum_all(self.data)
        out = NDTensor(total / n, (self,), 'mean')

        def _backward():
            scale = 1.0 / n
            self.grad = _map2(lambda g, s: g + s,
                              _map1(lambda _: out.grad * scale, self.data),
                              self.grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Activations
    # ------------------------------------------------------------------

    def tanh(self):
        """Element-wise tanh. Gradient: 1 - tanh(x)^2"""
        t = _map1(math.tanh, self.data)
        out = NDTensor(t, (self,), 'tanh')

        def _backward():
            # d/dx tanh = 1 - tanh(x)^2
            self.grad = _map2(
                lambda g, s: g + s,
                _map2(lambda g, tv: g * (1.0 - tv ** 2), out.grad, t),
                self.grad)

        out._backward = _backward
        return out

    def relu(self):
        """Element-wise relu. Gradient: 1 if x > 0 else 0"""
        t = _map1(lambda x: max(0.0, x), self.data)
        out = NDTensor(t, (self,), 'relu')

        def _backward():
            self.grad = _map2(
                lambda g, s: g + s,
                _map2(lambda g, x: g * (1.0 if x > 0 else 0.0),
                      out.grad, self.data),
                self.grad)

        out._backward = _backward
        return out

    def softmax(self):
        """
        Row-wise softmax for 2D tensor.
        shape (m, n) → (m, n), each row sums to 1.

        softmax(x)_i = exp(x_i) / sum(exp(x_j))

        Numerically stable: subtract row max before exp.
        Gradient: handled implicitly via cross-entropy pairing.
        """
        assert len(self.shape) == 2, "softmax requires 2D tensor"
        result = []
        for row in self.data:
            row_max = max(row)
            exps = [math.exp(x - row_max) for x in row]
            s = sum(exps)
            result.append([e / s for e in exps])

        out = NDTensor(result, (self,), 'softmax')

        def _backward():
            # Full Jacobian softmax gradient
            # dL/dx_i = s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
            grad = _zeros(self.shape)
            for i, (row_s, row_g) in enumerate(zip(out.data, out.grad)):
                dot = sum(g * s for g, s in zip(row_g, row_s))
                for j in range(len(row_s)):
                    grad[i][j] = row_s[j] * (row_g[j] - dot)
            self.grad = _map2(lambda g, s: g + s, grad, self.grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(self):
        """
        Reverse-mode automatic differentiation.
        Identical algorithm to scalar Tensor.backward().
        """
        topo    = []
        visited = set()

        def build_topo(t):
            if id(t) not in visited:
                visited.add(id(t))
                for parent in t._prev:
                    build_topo(parent)
                topo.append(t)

        build_topo(self)

        # Seed: scalar output gradient = 1
        if isinstance(self.data, float):
            self.grad = 1.0
        else:
            self.grad = _ones(self.shape)

        for t in reversed(topo):
            t._backward()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def zero_grad(self):
        self.grad = _zeros(self.shape) if self.shape else 0.0

    @staticmethod
    def zeros(shape):
        return NDTensor(_zeros(shape))

    @staticmethod
    def ones(shape):
        return NDTensor(_ones(shape))

    @staticmethod
    def randn(shape, scale=0.1):
        """
        Random normal initialization using Box-Muller transform.
        No numpy or random.gauss — pure Python.

        Box-Muller: given uniform u1, u2 in (0,1):
            z = sqrt(-2 * ln(u1)) * cos(2π * u2)
        produces standard normal samples.
        """
        import random

        def _randn_scalar():
            import random
            while True:
                u1 = random.random()
                u2 = random.random()
                if u1 > 0:
                    break
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            return z * scale

        def _fill(shape):
            if len(shape) == 1:
                return [_randn_scalar() for _ in range(shape[0])]
            return [_fill(shape[1:]) for _ in range(shape[0])]

        return NDTensor(_fill(shape))

    @staticmethod
    def from_list(data):
        """Convenience: wrap a nested Python list."""
        return NDTensor(data)

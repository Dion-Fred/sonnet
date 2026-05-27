"""
sonnet/tests/test_ndtensor.py

Tests for NDTensor and Linear layer.
Run from project root: python -m tests.test_ndtensor
"""

import sys
import os
import math
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sonnet_core.ndtensor import NDTensor, _matmul, _transpose, _zeros, _shape
from sonnet_core.linear import Linear, Sequential, ReLU, Tanh, MSELoss


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, condition, detail=''):
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f": {detail}"
    print(msg)
    return condition

def close(a, b, tol=1e-5):
    if isinstance(a, list) and isinstance(b, list):
        return all(close(ai, bi, tol) for ai, bi in zip(a, b))
    return abs(a - b) < tol


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def test_shape():
    print("\n--- test_shape ---")
    a = NDTensor([[1.0, 2.0], [3.0, 4.0]])
    check("2D shape", a.shape == (2, 2), str(a.shape))

    b = NDTensor([1.0, 2.0, 3.0])
    check("1D shape", b.shape == (3,), str(b.shape))

    c = NDTensor(5.0)
    check("scalar shape", c.shape == (), str(c.shape))


def test_zeros_ones():
    print("\n--- test_zeros_ones ---")
    z = NDTensor.zeros((2, 3))
    check("zeros shape", z.shape == (2, 3))
    check("zeros values", z.data == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    o = NDTensor.ones((2, 2))
    check("ones shape", o.shape == (2, 2))
    check("ones values", o.data == [[1.0, 1.0], [1.0, 1.0]])


# ------------------------------------------------------------------
# Element-wise ops
# ------------------------------------------------------------------

def test_add():
    print("\n--- test_add ---")
    A = NDTensor([[1.0, 2.0], [3.0, 4.0]])
    B = NDTensor([[1.0, 1.0], [1.0, 1.0]])
    C = A + B
    check("add forward", C.data == [[2.0, 3.0], [4.0, 5.0]])
    C.backward()
    check("A.grad", A.grad == [[1.0, 1.0], [1.0, 1.0]])
    check("B.grad", B.grad == [[1.0, 1.0], [1.0, 1.0]])


def test_sub():
    print("\n--- test_sub ---")
    A = NDTensor([[3.0, 4.0], [5.0, 6.0]])
    B = NDTensor([[1.0, 1.0], [1.0, 1.0]])
    C = A - B
    check("sub forward", C.data == [[2.0, 3.0], [4.0, 5.0]])
    C.backward()
    check("A.grad", A.grad == [[1.0, 1.0], [1.0, 1.0]])
    check("B.grad", B.grad == [[-1.0, -1.0], [-1.0, -1.0]])


def test_mul():
    print("\n--- test_mul ---")
    A = NDTensor([[2.0, 3.0]])
    B = NDTensor([[4.0, 5.0]])
    C = A * B
    check("mul forward", C.data == [[8.0, 15.0]])
    C.backward()
    check("A.grad", A.grad == [[4.0, 5.0]])
    check("B.grad", B.grad == [[2.0, 3.0]])


def test_pow():
    print("\n--- test_pow ---")
    A = NDTensor([[2.0, 3.0]])
    B = A ** 2
    check("pow forward", B.data == [[4.0, 9.0]])
    B.backward()
    # d(x^2)/dx = 2x
    check("A.grad", A.grad == [[4.0, 6.0]])


def test_neg():
    print("\n--- test_neg ---")
    A = NDTensor([[1.0, -2.0]])
    B = -A
    check("neg forward", B.data == [[-1.0, 2.0]])
    B.backward()
    check("A.grad", A.grad == [[-1.0, -1.0]])


# ------------------------------------------------------------------
# Transpose
# ------------------------------------------------------------------

def test_transpose():
    print("\n--- test_transpose ---")
    A = NDTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2×3
    At = A.T
    check("T shape", At.shape == (3, 2), str(At.shape))
    check("T data", At.data == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    At.backward()
    # grad of T is T of upstream grad (which is ones(3,2))
    check("A.grad shape", len(A.grad) == 2 and len(A.grad[0]) == 3)


# ------------------------------------------------------------------
# Matrix Multiply
# ------------------------------------------------------------------

def test_matmul_forward():
    print("\n--- test_matmul_forward ---")
    A = NDTensor([[1.0, 2.0], [3.0, 4.0]])   # 2×2
    B = NDTensor([[1.0, 0.0], [0.0, 1.0]])   # identity
    C = A @ B
    check("identity matmul", C.data == [[1.0, 2.0], [3.0, 4.0]])

    A2 = NDTensor([[1.0, 2.0]])     # 1×2
    B2 = NDTensor([[3.0], [4.0]])   # 2×1
    C2 = A2 @ B2
    check("dot product", C2.data == [[11.0]], str(C2.data))


def test_matmul_backward():
    print("\n--- test_matmul_backward ---")
    # Z = A @ B, sum all elements, backward
    # dL/dA = ones @ B.T,  dL/dB = A.T @ ones
    A = NDTensor([[1.0, 2.0], [3.0, 4.0]])   # 2×2
    B = NDTensor([[2.0, 0.0], [0.0, 2.0]])   # 2×2, diagonal

    Z = A @ B
    loss = Z.sum()
    loss.backward()

    # dL/dA[i][j] = sum_k(1 * B[j][k]) = sum of row j of B
    # B rows: [2,0] and [0,2], sums = 2 and 2
    # So each row of A.grad should be [2, 2]
    check("A.grad", A.grad == [[2.0, 2.0], [2.0, 2.0]], str(A.grad))

    # dL/dB[i][j] = sum_k(A[k][i]) = sum of col i of A
    # col 0 of A: 1+3=4, col 1: 2+4=6
    # B.grad should be [[4,4],[6,6]]... wait:
    # dL/dB = A.T @ dL/dZ, dL/dZ = ones(2,2)
    # A.T = [[1,3],[2,4]]
    # A.T @ ones(2,2) = [[1+3, 1+3],[2+4, 2+4]] = [[4,4],[6,6]]
    check("B.grad", B.grad == [[4.0, 4.0], [6.0, 6.0]], str(B.grad))


# ------------------------------------------------------------------
# Reductions
# ------------------------------------------------------------------

def test_sum():
    print("\n--- test_sum ---")
    A = NDTensor([[1.0, 2.0], [3.0, 4.0]])
    s = A.sum()
    check("sum forward", s.data == 10.0, str(s.data))
    s.backward()
    check("A.grad", A.grad == [[1.0, 1.0], [1.0, 1.0]])


def test_mean():
    print("\n--- test_mean ---")
    A = NDTensor([[1.0, 2.0], [3.0, 4.0]])
    m = A.mean()
    check("mean forward", m.data == 2.5, str(m.data))
    m.backward()
    check("A.grad", A.grad == [[0.25, 0.25], [0.25, 0.25]])


# ------------------------------------------------------------------
# Activations
# ------------------------------------------------------------------

def test_tanh():
    print("\n--- test_tanh ---")
    A = NDTensor([[0.0, 1.0]])
    B = A.tanh()
    check("tanh(0)=0", abs(B.data[0][0]) < 1e-9)
    check("tanh(1)", abs(B.data[0][1] - math.tanh(1.0)) < 1e-9)
    B.backward()
    # d/dx tanh at x=0 is 1, at x=1 is 1-tanh(1)^2
    expected_1 = 1.0 - math.tanh(1.0) ** 2
    check("tanh grad at 0", abs(A.grad[0][0] - 1.0) < 1e-9)
    check("tanh grad at 1", abs(A.grad[0][1] - expected_1) < 1e-9,
          f"got {A.grad[0][1]:.6f}")


def test_relu():
    print("\n--- test_relu ---")
    A = NDTensor([[-1.0, 0.0, 2.0]])
    B = A.relu()
    check("relu forward", B.data == [[0.0, 0.0, 2.0]])
    B.backward()
    check("relu grad", A.grad == [[0.0, 0.0, 1.0]])


# ------------------------------------------------------------------
# Broadcast add (bias)
# ------------------------------------------------------------------

def test_broadcast_add():
    print("\n--- test_broadcast_add (bias) ---")
    A = NDTensor([[1.0, 2.0], [3.0, 4.0]])   # (2, 2)
    b = NDTensor([10.0, 20.0])                 # (2,)
    C = A + b
    check("broadcast forward", C.data == [[11.0, 22.0], [13.0, 24.0]],
          str(C.data))
    C.backward()
    check("A.grad", A.grad == [[1.0, 1.0], [1.0, 1.0]])
    # bias grad = sum over rows: [2, 2]
    check("b.grad", b.grad == [2.0, 2.0], str(b.grad))


# ------------------------------------------------------------------
# Linear Layer
# ------------------------------------------------------------------

def test_linear_forward_shape():
    print("\n--- test_linear_forward_shape ---")
    random.seed(0)
    layer = Linear(4, 3)
    x = NDTensor([[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0]])   # batch=2, in=4
    out = layer(x)
    check("output shape", out.shape == (2, 3), str(out.shape))


def test_linear_parameter_count():
    print("\n--- test_linear_parameter_count ---")
    layer = Linear(4, 3)
    # W: (3, 4) = 12 values, b: (3,) = 3 values → 2 NDTensors
    check("parameter count", len(layer.parameters()) == 2)


def test_linear_backward():
    print("\n--- test_linear_backward ---")
    random.seed(1)
    layer = Linear(2, 2)
    x = NDTensor([[1.0, 0.0]])
    out = layer(x)
    loss = out.sum()
    loss.backward()
    # All W and b grads should be non-zero
    W_nonzero = any(layer.W.grad[i][j] != 0.0
                    for i in range(2) for j in range(2))
    b_nonzero = any(layer.b.grad[j] != 0.0 for j in range(2))
    check("W.grad non-zero", W_nonzero)
    check("b.grad non-zero", b_nonzero)


def test_sequential():
    print("\n--- test_sequential ---")
    random.seed(2)
    model = Sequential(
        Linear(2, 4),
        Tanh(),
        Linear(4, 1),
    )
    x = NDTensor([[1.0, -1.0]])
    out = model(x)
    check("output shape", out.shape == (1, 1), str(out.shape))
    loss = out.sum()
    loss.backward()
    for p in model.parameters():
        has_grad = any(
            p.grad[i][j] != 0.0
            for i in range(len(p.grad))
            for j in range(len(p.grad[0]) if isinstance(p.grad[0], list) else 1)
        ) if isinstance(p.grad, list) and isinstance(p.grad[0], list) else \
            any(g != 0.0 for g in p.grad) if isinstance(p.grad, list) else \
            p.grad != 0.0
        check(f"{p._label or 'param'}.grad non-zero", has_grad)


def test_training_step():
    print("\n--- test_training_step (loss decreases) ---")
    random.seed(42)
    model = Sequential(
        Linear(2, 4),
        Tanh(),
        Linear(4, 1),
    )
    loss_fn = MSELoss()

    # Single sample: input [1, 0] → target 1.0
    x      = NDTensor([[1.0, 0.0]])
    target = NDTensor([[1.0]])

    pred   = model(x)
    loss   = loss_fn(pred, target)
    loss_before = loss.data

    model.zero_grad()
    loss.backward()

    lr = 0.1
    for p in model.parameters():
        from sonnet_core.ndtensor import _map2
        if isinstance(p.data, list):
            for i in range(len(p.data)):
                if isinstance(p.data[i], list):
                    for j in range(len(p.data[i])):
                        p.data[i][j] -= lr * p.grad[i][j]
                else:
                    p.data[i] -= lr * p.grad[i]

    pred2  = model(x)
    loss2  = loss_fn(pred2, target)
    loss_after = loss2.data

    check("loss decreased", loss_after < loss_before,
          f"before={loss_before:.6f}, after={loss_after:.6f}")


# ------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_shape, test_zeros_ones,
        test_add, test_sub, test_mul, test_pow, test_neg,
        test_transpose,
        test_matmul_forward, test_matmul_backward,
        test_sum, test_mean,
        test_tanh, test_relu,
        test_broadcast_add,
        test_linear_forward_shape, test_linear_parameter_count,
        test_linear_backward, test_sequential, test_training_step,
    ]
    for t in tests:
        t()
    print("\nAll tests complete.")

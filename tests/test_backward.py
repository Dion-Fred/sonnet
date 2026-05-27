"""
sonnet/tests/test_backward.py

Tests for Sonnet autograd backward propagation.
Run from project root: python -m tests.test_backward
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sonnet_core.tensor import Tensor


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, got, expected, tol=1e-6):
    ok = abs(got - expected) < tol
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}: got={got:.6f}, expected={expected:.6f}")
    return ok


# ------------------------------------------------------------------

def test_mul():
    print("\n--- test_mul ---")
    x = Tensor(2.0)
    y = Tensor(3.0)
    z = x * y       # z = 6
    z.backward()
    # dz/dx = y = 3,  dz/dy = x = 2
    check("x.grad", x.grad, 3.0)
    check("y.grad", y.grad, 2.0)


def test_add():
    print("\n--- test_add ---")
    x = Tensor(4.0)
    y = Tensor(5.0)
    z = x + y       # z = 9
    z.backward()
    # dz/dx = 1,  dz/dy = 1
    check("x.grad", x.grad, 1.0)
    check("y.grad", y.grad, 1.0)


def test_sub():
    print("\n--- test_sub ---")
    x = Tensor(7.0)
    y = Tensor(3.0)
    z = x - y       # z = 4
    z.backward()
    # dz/dx = 1,  dz/dy = -1
    check("x.grad", x.grad, 1.0)
    check("y.grad", y.grad, -1.0)


def test_div():
    print("\n--- test_div ---")
    x = Tensor(6.0)
    y = Tensor(2.0)
    z = x / y       # z = 3
    z.backward()
    # dz/dx = 1/y = 0.5
    # dz/dy = -x/y^2 = -6/4 = -1.5
    check("x.grad", x.grad, 0.5)
    check("y.grad", y.grad, -1.5)


def test_pow():
    print("\n--- test_pow ---")
    x = Tensor(3.0)
    z = x ** 2      # z = 9
    z.backward()
    # dz/dx = 2x = 6
    check("x.grad", x.grad, 6.0)


def test_chain_rule():
    print("\n--- test_chain_rule (z = (x+y)*w) ---")
    x = Tensor(2.0)
    y = Tensor(3.0)
    w = Tensor(4.0)
    # Forward
    s = x + y       # s = 5
    z = s * w       # z = 20
    z.backward()
    # dz/ds = w = 4
    # dz/dx = dz/ds * ds/dx = 4*1 = 4
    # dz/dy = dz/ds * ds/dy = 4*1 = 4
    # dz/dw = s = 5
    check("x.grad", x.grad, 4.0)
    check("y.grad", y.grad, 4.0)
    check("w.grad", w.grad, 5.0)


def test_reuse_same_tensor():
    print("\n--- test_reuse_same_tensor (z = x*x) ---")
    x = Tensor(3.0)
    z = x * x       # z = x^2
    z.backward()
    # dz/dx = 2x = 6
    # Grad accumulates twice (once per edge): 3 + 3 = 6
    check("x.grad", x.grad, 6.0)


def test_scalar_operands():
    print("\n--- test_scalar_operands (Python scalars) ---")
    x = Tensor(5.0)
    z = x * 3       # __rmul__
    z.backward()
    check("x.grad", x.grad, 3.0)

    x2 = Tensor(4.0)
    z2 = 2 + x2     # __radd__
    z2.backward()
    check("x2.grad", x2.grad, 1.0)


def test_neg():
    print("\n--- test_neg ---")
    x = Tensor(5.0)
    z = -x
    z.backward()
    check("x.grad", x.grad, -1.0)


def test_quadratic():
    print("\n--- test_quadratic (L = (x-2)^2) ---")
    x = Tensor(5.0)
    # L = (x-2)^2 = x^2 - 4x + 4
    # dL/dx = 2(x-2) = 2*3 = 6
    diff = x - 2.0
    L = diff ** 2
    L.backward()
    check("x.grad", x.grad, 6.0)


# ------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_mul, test_add, test_sub, test_div, test_pow,
        test_chain_rule, test_reuse_same_tensor,
        test_scalar_operands, test_neg, test_quadratic,
    ]
    for t in tests:
        t()
    print("\nAll tests complete.")

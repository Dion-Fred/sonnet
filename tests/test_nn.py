"""
sonnet/tests/test_nn.py

Tests for Sonnet neural network components.
Run from project root: python -m tests.test_nn
"""

import sys
import os
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sonnet_core.tensor import Tensor
from sonnet_core.nn import Neuron, Layer, MLP, tanh, relu

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, condition, detail=''):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    return condition


# ------------------------------------------------------------------

def test_tanh_activation():
    print("\n--- test_tanh_activation ---")
    import math
    x = Tensor(0.5)
    out = tanh(x)
    check("tanh forward", abs(out.data - math.tanh(0.5)) < 1e-9)
    out.backward()
    expected_grad = 1.0 - math.tanh(0.5)**2
    check("tanh backward", abs(x.grad - expected_grad) < 1e-9,
          f"got {x.grad:.6f}, expected {expected_grad:.6f}")


def test_relu_activation():
    print("\n--- test_relu_activation ---")
    x_pos = Tensor(3.0)
    out = relu(x_pos)
    check("relu forward positive", out.data == 3.0)
    out.backward()
    check("relu backward positive", x_pos.grad == 1.0)

    x_neg = Tensor(-2.0)
    out2 = relu(x_neg)
    check("relu forward negative", out2.data == 0.0)
    out2.backward()
    check("relu backward negative", x_neg.grad == 0.0)


def test_neuron_output_is_tensor():
    print("\n--- test_neuron_output_is_tensor ---")
    random.seed(42)
    n = Neuron(3)
    out = n([1.0, 2.0, 3.0])
    check("output is Tensor", isinstance(out, Tensor))
    check("output in tanh range", -1.0 < out.data < 1.0,
          f"got {out.data:.6f}")


def test_neuron_parameter_count():
    print("\n--- test_neuron_parameter_count ---")
    n = Neuron(5)
    params = n.parameters()
    # 5 weights + 1 bias = 6
    check("parameter count", len(params) == 6, f"got {len(params)}")


def test_neuron_backward():
    print("\n--- test_neuron_backward ---")
    random.seed(0)
    n = Neuron(2)
    x = [Tensor(1.0), Tensor(2.0)]
    out = n(x)
    out.backward()
    # All parameters and inputs should have non-zero grad
    for i, p in enumerate(n.parameters()):
        check(f"param[{i}].grad != 0", p.grad != 0.0, f"got {p.grad:.6f}")


def test_neuron_zero_grad():
    print("\n--- test_neuron_zero_grad ---")
    random.seed(0)
    n = Neuron(2)
    out = n([1.0, 2.0])
    out.backward()
    n.zero_grad()
    for i, p in enumerate(n.parameters()):
        check(f"param[{i}].grad == 0 after zero_grad", p.grad == 0.0)


def test_layer_output_count():
    print("\n--- test_layer_output_count ---")
    random.seed(1)
    l = Layer(3, 4)
    out = l([1.0, 2.0, 3.0])
    check("output list length", len(out) == 4, f"got {len(out)}")
    check("outputs are Tensors", all(isinstance(o, Tensor) for o in out))


def test_layer_parameter_count():
    print("\n--- test_layer_parameter_count ---")
    l = Layer(3, 4)
    # Each neuron: 3 weights + 1 bias = 4 params, × 4 neurons = 16
    check("parameter count", len(l.parameters()) == 16,
          f"got {len(l.parameters())}")


def test_mlp_forward():
    print("\n--- test_mlp_forward ---")
    random.seed(7)
    model = MLP(2, [3, 1])
    out = model([1.0, 2.0])
    check("output is Tensor", isinstance(out, Tensor))
    check("output is scalar", isinstance(out.data, float))


def test_mlp_parameter_count():
    print("\n--- test_mlp_parameter_count ---")
    # MLP(2, [3, 1]):
    # Layer 0: 3 neurons, each with 2 inputs → 3*(2+1) = 9
    # Layer 1: 1 neuron, with 3 inputs       → 1*(3+1) = 4
    # Total: 13
    model = MLP(2, [3, 1])
    check("parameter count", len(model.parameters()) == 13,
          f"got {len(model.parameters())}")


def test_mlp_backward():
    print("\n--- test_mlp_backward ---")
    random.seed(3)
    model = MLP(2, [3, 1])
    out = model([0.5, -0.5])
    out.backward()
    for i, p in enumerate(model.parameters()):
        check(f"param[{i}].grad != 0", p.grad != 0.0, f"grad={p.grad:.6f}")


def test_mlp_training_step():
    """
    Minimal training loop: one input, one target, MSE loss.
    Checks that loss decreases after a gradient descent step.
    """
    print("\n--- test_mlp_training_step ---")
    random.seed(99)
    model = MLP(2, [4, 1])
    x = [1.0, -1.0]
    target = Tensor(1.0)

    # Forward
    pred = model(x)
    # MSE loss: (pred - target)^2
    loss = (pred - target) ** 2
    loss_before = loss.data

    # Backward
    model.zero_grad()
    loss.backward()

    # Gradient descent step
    lr = 0.1
    for p in model.parameters():
        p.data -= lr * p.grad

    # Forward again
    pred2 = model(x)
    loss2 = (pred2 - target) ** 2
    loss_after = loss2.data

    check("loss decreased", loss_after < loss_before,
          f"before={loss_before:.6f}, after={loss_after:.6f}")


# ------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_tanh_activation,
        test_relu_activation,
        test_neuron_output_is_tensor,
        test_neuron_parameter_count,
        test_neuron_backward,
        test_neuron_zero_grad,
        test_layer_output_count,
        test_layer_parameter_count,
        test_mlp_forward,
        test_mlp_parameter_count,
        test_mlp_backward,
        test_mlp_training_step,
    ]
    for t in tests:
        t()
    print("\nAll tests complete.")

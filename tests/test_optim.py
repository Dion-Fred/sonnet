"""
sonnet/tests/test_optim.py

Tests for Sonnet optimizers.
Run from project root: python -m tests.test_optim
"""

import sys
import os
import math
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sonnet_core.ndtensor import NDTensor
from sonnet_core.linear import Linear, Sequential, Tanh, MSELoss
from sonnet_core.optim import SGD, Adam


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, condition, detail=''):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    return condition


def make_quadratic(w_init, target):
    """L = (w - target)^2, minimum at w=target."""
    w = NDTensor([[w_init]])
    def loss_fn():
        t = NDTensor([[target]])
        diff = w - t
        return (diff ** 2).sum()
    return w, loss_fn


# ------------------------------------------------------------------
# SGD
# ------------------------------------------------------------------

def test_sgd_plain_loss_decreases():
    print("\n--- test_sgd_plain (loss decreases) ---")
    random.seed(0)
    model = Sequential(Linear(2, 4), Tanh(), Linear(4, 1))
    loss_fn = MSELoss()
    opt = SGD(model.parameters(), lr=0.05)
    x = NDTensor([[1.0, -1.0]])
    y = NDTensor([[1.0]])
    losses = []
    for _ in range(10):
        pred = model(x)
        loss = loss_fn(pred, y)
        losses.append(loss.data)
        opt.zero_grad()
        loss.backward()
        opt.step()
    check("loss decreases over 10 steps",
          losses[-1] < losses[0],
          f"start={losses[0]:.4f}, end={losses[-1]:.4f}")


def test_sgd_converges_quadratic():
    print("\n--- test_sgd_converges_quadratic ---")
    w, loss_fn = make_quadratic(0.0, 3.0)
    opt = SGD([w], lr=0.1)
    for _ in range(50):
        loss = loss_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
    check("w converges to 3.0",
          abs(w.data[0][0] - 3.0) < 0.01,
          f"got {w.data[0][0]:.4f}")


def test_sgd_momentum_faster_than_plain():
    """
    Momentum wins when lr is deliberately small.

    At lr=0.005, plain SGD makes tiny steps and barely moves.
    Momentum accumulates velocity over steps and reaches a much
    lower loss in the same number of steps.

    This is the honest test: same lr for both, but small enough
    that momentum's accumulation effect is the decisive factor.
    At a well-tuned lr, plain SGD can match or beat momentum on
    simple surfaces — that is expected and correct behavior.
    """
    print("\n--- test_sgd_momentum_faster_than_plain ---")

    def run(momentum, steps=80):
        random.seed(5)
        model = Sequential(Linear(2, 8), Tanh(), Linear(8, 1))
        loss_fn = MSELoss()
        opt = SGD(model.parameters(), lr=0.005, momentum=momentum)
        x = NDTensor([[1.0, -1.0]])
        y = NDTensor([[1.0]])
        for _ in range(steps):
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.data

    plain_loss    = run(momentum=0.0)
    momentum_loss = run(momentum=0.9)
    check("momentum reaches lower loss than plain SGD at small lr",
          momentum_loss < plain_loss,
          f"plain={plain_loss:.4f}, momentum={momentum_loss:.4f}")


def test_sgd_zero_grad_clears():
    print("\n--- test_sgd_zero_grad_clears ---")
    random.seed(0)
    model = Sequential(Linear(2, 2), Tanh(), Linear(2, 1))
    opt = SGD(model.parameters(), lr=0.01)
    x = NDTensor([[1.0, 0.0]])
    pred = model(x)
    pred.sum().backward()
    opt.zero_grad()
    all_zero = True
    for p in model.parameters():
        if isinstance(p.grad, list):
            for row in p.grad:
                if isinstance(row, list):
                    for g in row:
                        if g != 0.0: all_zero = False
                else:
                    if row != 0.0: all_zero = False
        else:
            if p.grad != 0.0: all_zero = False
    check("all grads zero after zero_grad()", all_zero)


# ------------------------------------------------------------------
# Adam
# ------------------------------------------------------------------

def test_adam_loss_decreases():
    print("\n--- test_adam_loss_decreases ---")
    random.seed(1)
    model = Sequential(Linear(2, 4), Tanh(), Linear(4, 1))
    loss_fn = MSELoss()
    opt = Adam(model.parameters(), lr=1e-2)
    x = NDTensor([[1.0, -1.0]])
    y = NDTensor([[1.0]])
    losses = []
    for _ in range(10):
        pred = model(x)
        loss = loss_fn(pred, y)
        losses.append(loss.data)
        opt.zero_grad()
        loss.backward()
        opt.step()
    check("loss decreases over 10 steps",
          losses[-1] < losses[0],
          f"start={losses[0]:.4f}, end={losses[-1]:.4f}")


def test_adam_converges_quadratic():
    print("\n--- test_adam_converges_quadratic ---")
    w, loss_fn = make_quadratic(0.0, 5.0)
    opt = Adam([w], lr=0.1)
    for _ in range(100):
        loss = loss_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
    check("w converges to 5.0",
          abs(w.data[0][0] - 5.0) < 0.05,
          f"got {w.data[0][0]:.4f}")


def test_adam_faster_than_sgd():
    print("\n--- test_adam_faster_than_sgd ---")

    def run_sgd(steps=50):
        random.seed(9)
        model = Sequential(Linear(3, 8), Tanh(), Linear(8, 1))
        loss_fn = MSELoss()
        opt = SGD(model.parameters(), lr=0.01)
        x = NDTensor([[1.0, -0.5, 0.3]])
        y = NDTensor([[1.0]])
        for _ in range(steps):
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.data

    def run_adam(steps=50):
        random.seed(9)
        model = Sequential(Linear(3, 8), Tanh(), Linear(8, 1))
        loss_fn = MSELoss()
        opt = Adam(model.parameters(), lr=1e-2)
        x = NDTensor([[1.0, -0.5, 0.3]])
        y = NDTensor([[1.0]])
        for _ in range(steps):
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.data

    sgd_loss  = run_sgd()
    adam_loss = run_adam()
    check("Adam reaches lower loss than SGD in 50 steps",
          adam_loss < sgd_loss,
          f"SGD={sgd_loss:.4f}, Adam={adam_loss:.4f}")


def test_adam_step_counter():
    print("\n--- test_adam_step_counter ---")
    random.seed(0)
    model = Sequential(Linear(2, 2))
    opt = Adam(model.parameters())
    x = NDTensor([[1.0, 0.0]])
    for i in range(5):
        out = model(x)
        out.sum().backward()
        opt.step()
        opt.zero_grad()
    check("step counter correct", opt.t == 5, str(opt.t))


def test_adam_bias_correction_early_steps():
    print("\n--- test_adam_bias_correction_early_steps ---")
    w, loss_fn = make_quadratic(0.0, 1.0)
    opt = Adam([w], lr=0.1)
    w_before = w.data[0][0]
    loss = loss_fn()
    opt.zero_grad()
    loss.backward()
    opt.step()
    w_after = w.data[0][0]
    update_magnitude = abs(w_after - w_before)
    check("first-step update is substantial (bias correction working)",
          update_magnitude > 0.05,
          f"update={update_magnitude:.6f}")


def test_weight_decay():
    print("\n--- test_weight_decay (SGD) ---")

    def run(weight_decay, steps=20):
        random.seed(3)
        model = Sequential(Linear(2, 4), Tanh(), Linear(4, 1))
        opt = SGD(model.parameters(), lr=0.01, weight_decay=weight_decay)
        loss_fn = MSELoss()
        x = NDTensor([[2.0, 2.0]])
        y = NDTensor([[0.0]])
        for _ in range(steps):
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        total = 0.0
        for p in model.parameters():
            if isinstance(p.data, list):
                for row in p.data:
                    if isinstance(row, list):
                        for v in row: total += abs(v)
                    else:
                        total += abs(row)
        return total

    no_decay   = run(weight_decay=0.0)
    with_decay = run(weight_decay=0.01)
    check("weight decay reduces total weight magnitude",
          with_decay < no_decay,
          f"no_decay={no_decay:.4f}, with_decay={with_decay:.4f}")


# ------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_sgd_plain_loss_decreases,
        test_sgd_converges_quadratic,
        test_sgd_momentum_faster_than_plain,
        test_sgd_zero_grad_clears,
        test_adam_loss_decreases,
        test_adam_converges_quadratic,
        test_adam_faster_than_sgd,
        test_adam_step_counter,
        test_adam_bias_correction_early_steps,
        test_weight_decay,
    ]
    for t in tests:
        t()
    print("\nAll tests complete.")

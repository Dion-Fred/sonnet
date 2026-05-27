"""
sonnet/sonnet_core/optim.py

Sonnet Optimizers

Replaces the raw parameter update loop with proper optimizer objects.

Available:
    SGD    — Stochastic Gradient Descent with optional momentum
    Adam   — Adaptive Moment Estimation

All optimizers share the same interface:
    optimizer.step()       — update parameters using stored gradients
    optimizer.zero_grad()  — reset all gradients to zero

Usage:
    optimizer = Adam(model.parameters(), lr=1e-3)

    for step in range(steps):
        pred  = model(x)
        loss  = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""

import math
from sonnet_core.ndtensor import NDTensor, _zeros, _map2, _map1


# ------------------------------------------------------------------
# Utility: apply a function to every scalar in a nested structure,
# updating in-place. Used for parameter and state mutation.
# ------------------------------------------------------------------

def _apply_inplace(data, grad, fn):
    """
    Traverse data and grad (same shape) simultaneously.
    Apply fn(d, g) → new value, write back into data.
    Mutates data in-place.
    """
    if not isinstance(data, list):
        # base case: caller handles scalar replacement
        return fn(data, grad)
    for i in range(len(data)):
        if isinstance(data[i], list):
            _apply_inplace(data[i], grad[i], fn)
        else:
            data[i] = fn(data[i], grad[i])


def _apply_state_inplace(state, fn):
    """Apply fn(scalar) → scalar across a nested list. Mutates in-place."""
    if not isinstance(state, list):
        return fn(state)
    for i in range(len(state)):
        if isinstance(state[i], list):
            _apply_state_inplace(state[i], fn)
        else:
            state[i] = fn(state[i])


def _map2_inplace(target, a, b, fn):
    """target[...] = fn(a[...], b[...]) element-wise. Mutates target."""
    if not isinstance(target, list):
        return fn(a, b)
    for i in range(len(target)):
        if isinstance(target[i], list):
            _map2_inplace(target[i], a[i], b[i], fn)
        else:
            target[i] = fn(a[i], b[i])


def _map3_inplace(target, a, b, c, fn):
    """target[...] = fn(a[...], b[...], c[...]) element-wise."""
    if not isinstance(target, list):
        return fn(a, b, c)
    for i in range(len(target)):
        if isinstance(target[i], list):
            _map3_inplace(target[i], a[i], b[i], c[i], fn)
        else:
            target[i] = fn(a[i], b[i], c[i])


def _deep_copy_structure(data):
    """Create a zero-filled structure matching the shape of data."""
    if not isinstance(data, list):
        return 0.0
    return [_deep_copy_structure(d) for d in data]


# ------------------------------------------------------------------
# Base Optimizer
# ------------------------------------------------------------------

class Optimizer:
    """
    Base class for all Sonnet optimizers.

    Subclasses must implement step().
    """

    def __init__(self, parameters):
        """
        parameters: list of NDTensor (from model.parameters())
        """
        self.parameters = list(parameters)

    def zero_grad(self):
        """Reset all parameter gradients to zero."""
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        raise NotImplementedError("Subclasses must implement step()")


# ------------------------------------------------------------------
# SGD with Momentum
# ------------------------------------------------------------------

class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.

    Update rule (no momentum, β=0):
        p = p - lr * grad

    Update rule (with momentum):
        v = β * v + (1 - β) * grad
        p = p - lr * v

    The velocity v accumulates a smoothed estimate of gradient direction.
    Parameters that have been consistently moving in the same direction
    build up speed. Oscillating gradients cancel each other out.

    Parameters:
        parameters  — list of NDTensor from model.parameters()
        lr          — learning rate (default: 0.01)
        momentum    — β coefficient (default: 0.0 = plain SGD)
        weight_decay— L2 regularization coefficient (default: 0.0)

    Typical values:
        lr=0.01, momentum=0.9   — standard choice for most tasks
        lr=0.1,  momentum=0.0   — plain SGD for simple problems
    """

    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters)
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay

        # Velocity buffers — one per parameter, same shape
        # Initialized to zero; built up over training steps
        self.velocities = [
            _deep_copy_structure(p.data) for p in self.parameters
        ]

    def step(self):
        """
        Perform one parameter update using current gradients.
        Call after loss.backward().
        """
        for p, v in zip(self.parameters, self.velocities):
            grad = p.grad

            # L2 weight decay: effectively adds λ*p to the gradient
            # This penalizes large weights, reduces overfitting
            if self.weight_decay != 0.0:
                grad = _map2(
                    lambda g, d: g + self.weight_decay * d,
                    grad, p.data)

            if self.momentum != 0.0:
                # v = β*v + (1-β)*grad
                beta = self.momentum
                _map2_inplace(v, v, grad,
                              lambda vi, gi: beta * vi + (1 - beta) * gi)
                # p = p - lr * v
                _apply_inplace(p.data, v,
                               lambda d, vi: d - self.lr * vi)
            else:
                # Plain SGD: p = p - lr * grad
                _apply_inplace(p.data, grad,
                               lambda d, g: d - self.lr * g)

    def __repr__(self):
        return (f"SGD(lr={self.lr}, momentum={self.momentum}, "
                f"weight_decay={self.weight_decay})")


# ------------------------------------------------------------------
# Adam
# ------------------------------------------------------------------

class Adam(Optimizer):
    """
    Adam: Adaptive Moment Estimation.
    Kingma & Ba, 2014 (https://arxiv.org/abs/1412.6980)

    Update rule:
        m = β1*m + (1-β1)*grad               # first moment (mean)
        v = β2*v + (1-β2)*grad²              # second moment (variance)

        m̂ = m / (1 - β1^t)                  # bias correction
        v̂ = v / (1 - β2^t)                  # bias correction

        p = p - lr * m̂ / (sqrt(v̂) + ε)

    Why bias correction?
        At step 1, m and v are initialized to zero.
        Without correction: m̂ = (1-β1)*grad ≈ 0.1*grad (too small).
        Dividing by (1-β1^1) = 0.1 restores the true scale.
        As t grows, β1^t → 0 and the correction approaches 1 (no effect).

    Why divide by sqrt(v̂)?
        v̂ tracks the mean squared gradient magnitude per parameter.
        Dividing normalizes updates: parameters with large historical
        gradients get smaller steps; sparse parameters get larger steps.
        This makes Adam robust to different scales across parameters.

    Parameters:
        parameters  — list of NDTensor
        lr          — learning rate (default: 1e-3)
        beta1       — first moment decay  (default: 0.9)
        beta2       — second moment decay (default: 0.999)
        eps         — numerical stability (default: 1e-8)
        weight_decay— L2 regularization  (default: 0.0)

    Defaults work well for almost all tasks.
    The main thing to tune is lr (try 1e-4, 1e-3, 3e-3).
    """

    def __init__(self, parameters, lr=1e-3,
                 beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(parameters)
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay

        # Step counter — needed for bias correction
        self.t = 0

        # First moment (mean of gradients) — same shape as each parameter
        self.m = [_deep_copy_structure(p.data) for p in self.parameters]

        # Second moment (mean of squared gradients)
        self.v = [_deep_copy_structure(p.data) for p in self.parameters]

    def step(self):
        """
        Perform one Adam parameter update.
        Call after loss.backward().
        """
        self.t += 1
        t = self.t

        # Bias correction denominators
        bc1 = 1.0 - self.beta1 ** t
        bc2 = 1.0 - self.beta2 ** t

        for p, m, v in zip(self.parameters, self.m, self.v):
            grad = p.grad

            # Weight decay
            if self.weight_decay != 0.0:
                grad = _map2(
                    lambda g, d: g + self.weight_decay * d,
                    grad, p.data)

            # Update first moment: m = β1*m + (1-β1)*grad
            _map2_inplace(m, m, grad,
                          lambda mi, gi: self.beta1 * mi + (1 - self.beta1) * gi)

            # Update second moment: v = β2*v + (1-β2)*grad²
            _map2_inplace(v, v, grad,
                          lambda vi, gi: self.beta2 * vi + (1 - self.beta2) * gi * gi)

            # Apply bias-corrected update: p = p - lr * m̂ / (sqrt(v̂) + ε)
            eps   = self.eps
            lr    = self.lr

            def _adam_update(pi, mi, vi):
                m_hat = mi / bc1
                v_hat = vi / bc2
                return pi - lr * m_hat / (math.sqrt(v_hat) + eps)

            _map3_inplace(p.data, p.data, m, v, _adam_update)

    def __repr__(self):
        return (f"Adam(lr={self.lr}, β1={self.beta1}, "
                f"β2={self.beta2}, ε={self.eps})")

"""
sonnet/sonnet_core/linear.py

Sonnet Linear Layer — matrix operations via NDTensor

This replaces the scalar Neuron/Layer for serious computation.
A Linear layer computes:

    output = input @ W.T + b

Where:
    input : NDTensor of shape (batch, in_features)
    W     : NDTensor of shape (out_features, in_features)  — weight matrix
    b     : NDTensor of shape (out_features,)               — bias vector
    output: NDTensor of shape (batch, out_features)

Why W.T?
    Convention: W stores one output neuron per row.
    input @ W.T produces (batch, out_features) — one prediction per sample.
    This matches PyTorch's nn.Linear convention exactly.

Weight initialization:
    Kaiming (He) uniform initialization:
        scale = sqrt(2 / in_features)
    Designed for relu activations. Keeps variance stable through layers.
    Without careful init, signals either vanish or explode in deep networks.
"""

import math
import random

from sonnet_core.ndtensor import NDTensor, _zeros


class Module:
    """Base class — same interface as scalar nn.Module."""

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


# ------------------------------------------------------------------
# Activation functions (operate on NDTensor)
# ------------------------------------------------------------------

def relu(x):
    return x.relu()

def tanh(x):
    return x.tanh()

def softmax(x):
    return x.softmax()


# ------------------------------------------------------------------
# Linear Layer
# ------------------------------------------------------------------

class Linear(Module):
    """
    Fully connected linear transformation.

    output = input @ W.T + b

    Parameters:
        in_features  — size of each input sample
        out_features — size of each output sample
        bias         — if True (default), adds a learnable bias

    Initialization:
        Kaiming uniform: W ~ Uniform(-k, k) where k = sqrt(1/in_features)
        b = zeros
    """

    def __init__(self, in_features, out_features, bias=True):
        self.in_features  = in_features
        self.out_features = out_features
        self.use_bias     = bias

        # Kaiming uniform scale
        k = math.sqrt(1.0 / in_features)

        # W shape: (out_features, in_features)
        self.W = NDTensor([
            [random.uniform(-k, k) for _ in range(in_features)]
            for _ in range(out_features)
        ])
        self.W._label = 'W'

        # b shape: (out_features,)
        self.b = NDTensor([0.0] * out_features) if bias else None
        if self.b:
            self.b._label = 'b'

    def forward(self, x):
        """
        x: NDTensor of shape (batch, in_features)
           or (in_features,) — single sample, auto-promoted to (1, in_features)

        Returns NDTensor of shape (batch, out_features).
        """
        # Promote 1D input to 2D batch of 1
        if len(x.shape) == 1:
            x = NDTensor([x.data])   # (1, in_features)

        # (batch, in_features) @ (in_features, out_features) = (batch, out_features)
        # W is (out_features, in_features), so we need W.T
        from sonnet_core.ndtensor import _transpose, NDTensor as _NDT
        out = x @ NDTensor([[self.W.data[j][i]
                              for j in range(self.out_features)]
                             for i in range(self.in_features)])

        # Actually: let's use the proper W.T property
        out = x @ self.W.T

        if self.use_bias:
            out = out + self.b   # broadcast add: (batch, out) + (out,)

        return out

    def parameters(self):
        params = [self.W]
        if self.use_bias:
            params.append(self.b)
        return params

    def __repr__(self):
        return (f"Linear(in={self.in_features}, "
                f"out={self.out_features}, bias={self.use_bias})")


# ------------------------------------------------------------------
# Sequential — chain of layers
# ------------------------------------------------------------------

class Sequential(Module):
    """
    Chains layers in order. Output of each becomes input of next.

    Usage:
        model = Sequential(
            Linear(2, 4),
            ReLU(),
            Linear(4, 1),
        )
        out = model(x)
    """

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self):
        layer_strs = '\n  '.join(str(l) for l in self.layers)
        total = len(self.parameters())
        return f"Sequential(\n  {layer_strs}\n) — {total} parameters"


# ------------------------------------------------------------------
# Activation wrapper layers (for use in Sequential)
# ------------------------------------------------------------------

class ReLU(Module):
    def forward(self, x):
        return x.relu()
    def __repr__(self):
        return "ReLU()"


class Tanh(Module):
    def forward(self, x):
        return x.tanh()
    def __repr__(self):
        return "Tanh()"


# ------------------------------------------------------------------
# Loss Functions (NDTensor versions)
# ------------------------------------------------------------------

class MSELoss(Module):
    """
    Mean Squared Error over a batch.

    predictions: NDTensor (batch, 1) or (batch,)
    targets:     NDTensor (batch, 1) or list of float

    L = mean( (pred - target)^2 )
    """

    def forward(self, predictions, targets):
        if not isinstance(targets, NDTensor):
            targets = NDTensor([[float(t)] for t in targets])

        diff = predictions - targets
        return (diff ** 2).mean()

    def __repr__(self):
        return "MSELoss()"

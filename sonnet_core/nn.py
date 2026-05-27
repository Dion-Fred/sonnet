"""
sonnet/sonnet_core/nn.py

Sonnet Neural Network Layer
Neuron → Layer → MLP

Built entirely on sonnet_core.tensor.Tensor.
No external dependencies.
"""

import math
import random
from sonnet_core.tensor import Tensor


# ------------------------------------------------------------------
# Activation Functions
# ------------------------------------------------------------------

def tanh(x):
    """
    Hyperbolic tangent activation.

    Formula:   tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    Output:    (-1, 1)
    Gradient:  d/dx tanh(x) = 1 - tanh(x)^2

    We implement this directly on Tensor using existing operations
    so the autograd graph is built automatically.
    """
    # tanh(x) = (e^2x - 1) / (e^2x + 1)
    # Using exp from math — we'll add Tensor.exp() shortly;
    # for now compute via Python float and wrap back in Tensor.
    t = math.tanh(x.data)
    out = Tensor(t, (x,), 'tanh')

    # d/dx tanh = 1 - tanh(x)^2
    def _backward():
        x.grad += (1.0 - t ** 2) * out.grad

    out._backward = _backward
    return out


def relu(x):
    """
    Rectified Linear Unit.

    Formula:   relu(x) = max(0, x)
    Output:    [0, ∞)
    Gradient:  1 if x > 0 else 0

    Simpler than tanh but suffers from dying neurons at x <= 0.
    """
    t = max(0.0, x.data)
    out = Tensor(t, (x,), 'relu')

    def _backward():
        x.grad += (1.0 if x.data > 0 else 0.0) * out.grad

    out._backward = _backward
    return out


# ------------------------------------------------------------------
# Module Base Class
# ------------------------------------------------------------------

class Module:
    """
    Base class for all Sonnet neural network components.

    Provides:
    - parameters() — returns all Tensor weights and biases
    - zero_grad()  — resets all gradients before each backward pass
    """

    def parameters(self):
        """Override in subclasses to return list of Tensor parameters."""
        return []

    def zero_grad(self):
        """Reset all parameter gradients to zero."""
        for p in self.parameters():
            p.grad = 0.0

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward()")


# ------------------------------------------------------------------
# Neuron
# ------------------------------------------------------------------

class Neuron(Module):
    """
    A single artificial neuron.

    Computes:
        output = activation( sum(wi * xi for each input) + b )

    Parameters:
        nin         — number of inputs
        activation  — 'tanh' (default) or 'relu' or None (linear)

    Weights are initialized with small random values from [-1, 1].
    Bias is initialized to 0.

    Why random weights?
    If all weights start at the same value, all neurons in a layer
    compute identical outputs and receive identical gradients.
    They never differentiate. This is called the symmetry problem.
    Random initialization breaks symmetry.

    Why small values?
    Large initial weights push activations into saturation zones
    where gradients are near zero (vanishing gradient problem).
    """

    def __init__(self, nin, activation='tanh'):
        # One weight per input, randomly initialized
        self.w = [Tensor(random.uniform(-1, 1)) for _ in range(nin)]
        # Bias starts at zero
        self.b = Tensor(0.0)
        self.activation = activation

    def forward(self, x):
        """
        x: list of Tensor or list of float/int

        Returns a single Tensor — the neuron's output.
        """
        # Ensure inputs are Tensors
        inputs = [xi if isinstance(xi, Tensor) else Tensor(xi) for xi in x]

        # Weighted sum: w0*x0 + w1*x1 + ... + wn*xn + b
        # Start from bias, accumulate products
        act = self.b
        for wi, xi in zip(self.w, inputs):
            act = act + wi * xi

        # Apply activation
        if self.activation == 'tanh':
            return tanh(act)
        elif self.activation == 'relu':
            return relu(act)
        elif self.activation is None:
            return act   # linear neuron
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def parameters(self):
        # Weights + bias = nin + 1 parameters per neuron
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron(nin={len(self.w)}, activation='{self.activation}')"


# ------------------------------------------------------------------
# Layer
# ------------------------------------------------------------------

class Layer(Module):
    """
    A fully connected layer: a list of neurons, each receiving
    the same inputs and producing one output each.

    Parameters:
        nin   — number of inputs to each neuron
        nout  — number of neurons in this layer
        **kwargs passed to each Neuron (e.g. activation)

    Output: list of nout Tensors
    """

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def forward(self, x):
        # Each neuron independently processes the full input vector
        return [n(x) for n in self.neurons]

    def parameters(self):
        # Flatten: all parameters from all neurons
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params

    def __repr__(self):
        return f"Layer([{', '.join(str(n) for n in self.neurons)}])"


# ------------------------------------------------------------------
# MLP — Multi-Layer Perceptron
# ------------------------------------------------------------------

class MLP(Module):
    """
    A multi-layer perceptron: a sequence of fully connected layers.

    Parameters:
        nin      — number of inputs
        nouts    — list of output sizes per layer
                   e.g. [4, 4, 1] → two hidden layers of 4, one output of 1

    The last layer uses linear (no) activation by convention
    when used for regression. For classification, tanh or sigmoid
    is appropriate on the output.

    Example:
        model = MLP(3, [4, 4, 1])
        # Input: 3 features
        # Hidden layer 1: 4 neurons, tanh
        # Hidden layer 2: 4 neurons, tanh
        # Output layer: 1 neuron, linear
    """

    def __init__(self, nin, nouts):
        # Build layer size pairs: (nin, nouts[0]), (nouts[0], nouts[1]), ...
        sizes = [nin] + nouts

        self.layers = []
        for i in range(len(nouts)):
            is_last = (i == len(nouts) - 1)
            activation = None if is_last else 'tanh'
            self.layers.append(Layer(sizes[i], sizes[i+1], activation=activation))

    def forward(self, x):
        """
        x: list of float/int/Tensor — the input vector

        Passes x through each layer in sequence.
        Each layer's output becomes the next layer's input.
        """
        out = x
        for layer in self.layers:
            out = layer(out)

        # If output layer has one neuron, unwrap the list
        if len(out) == 1:
            return out[0]
        return out

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self):
        layer_strs = '\n  '.join(str(l) for l in self.layers)
        total = len(self.parameters())
        return f"MLP(\n  {layer_strs}\n) — {total} parameters"

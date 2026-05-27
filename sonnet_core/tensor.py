"""
sonnet/sonnet_core/tensor.py

Sonnet Core Tensor — with Autograd and Backward Propagation
"""


class Tensor:
    """
    A scalar-valued tensor with:
    - forward computation
    - computational graph construction
    - backward propagation via chain rule
    """

    def __init__(self, data, _children=(), _op='', _label=''):
        self.data = float(data)   # always store as float for gradient math
        self.grad = 0.0           # gradient starts at zero

        # _backward holds the function that computes this tensor's
        # contribution to its children's gradients.
        # Default: no-op (leaf tensors have no parents to propagate to)
        self._backward = lambda: None

        self._prev = set(_children)   # parent tensors in the graph
        self._op = _op                # operation that created this tensor
        self._label = _label          # optional name, useful for debugging

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"


    # ------------------------------------------------------------------
    # Forward Operations
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, (self, other), '+')

        # d(out)/d(self)  = 1
        # d(out)/d(other) = 1
        # By chain rule:
        #   self.grad  += 1.0 * out.grad
        #   other.grad += 1.0 * out.grad
        def _backward():
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        # Handles: scalar + Tensor
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data - other.data, (self, other), '-')

        # d(out)/d(self)  = +1
        # d(out)/d(other) = -1
        def _backward():
            self.grad  += 1.0 * out.grad
            other.grad += -1.0 * out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other):
        # Handles: scalar - Tensor
        return Tensor(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, (self, other), '*')

        # d(out)/d(self)  = other.data
        # d(out)/d(other) = self.data
        # We capture current values via closure — critical detail.
        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        # Handles: scalar * Tensor
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data / other.data, (self, other), '/')

        # z = x / y  =>  z = x * y^(-1)
        # d(out)/d(self)  =  1 / other.data
        # d(out)/d(other) = -self.data / (other.data ** 2)
        def _backward():
            self.grad  += (1.0 / other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        # Handles: scalar / Tensor
        return Tensor(other) / self

    def __neg__(self):
        return self * -1

    def __pow__(self, exponent):
        # Only supports numeric exponents (int or float), not Tensor
        assert isinstance(exponent, (int, float)), "Exponent must be int or float"

        out = Tensor(self.data ** exponent, (self,), f'**{exponent}')

        # d(x^n)/dx = n * x^(n-1)
        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out


    # ------------------------------------------------------------------
    # Backward Propagation
    # ------------------------------------------------------------------

    def backward(self):
        """
        Compute gradients for all tensors in the computational graph
        via reverse-mode automatic differentiation.

        Algorithm:
        1. Topologically sort the graph (children before parents)
        2. Set this tensor's gradient to 1.0 (dL/dL = 1)
        3. Walk the graph in reverse order
        4. Call each tensor's _backward() to propagate gradients
        """

        # Step 1: Build topological order
        # Topological sort ensures every node is processed only after
        # all nodes that depend on it have already been processed.
        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._prev:
                    build_topo(parent)
                topo.append(tensor)   # append AFTER processing all parents

        build_topo(self)

        # Step 2: Seed gradient
        # The output tensor's gradient with respect to itself is 1.
        # This is the starting point: dL/dL = 1.
        self.grad = 1.0

        # Step 3 & 4: Walk reversed topological order, call _backward()
        # Reversed topo = output first, inputs last
        for tensor in reversed(topo):
            tensor._backward()


    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def zero_grad(self):
        """Reset gradient to zero. Used between training steps."""
        self.grad = 0.0
        
"""
sonnet/sonnet_core/loss.py

Sonnet Loss Functions

Loss functions measure how wrong the model is.
They produce a single scalar Tensor — the loss.
Calling loss.backward() propagates gradients through
the entire computational graph back to model parameters.

Available:
    MSELoss              — Mean Squared Error
    BinaryCrossEntropy   — Binary Cross-Entropy Loss
"""

import math
from sonnet_core.tensor import Tensor
from sonnet_core.nn import Module


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(float(x))


# ------------------------------------------------------------------
# MSE Loss
# ------------------------------------------------------------------

class MSELoss(Module):
    """
    Mean Squared Error Loss.

    Formula:
        L = (1/n) * sum( (pred_i - target_i)^2 )

    Gradient w.r.t. each prediction:
        dL/dpred_i = (2/n) * (pred_i - target_i)

    Use when:
        - Regression tasks (predicting continuous values)
        - You want large errors penalized quadratically

    Weakness:
        - Outliers dominate (large errors squared become huge)
        - Not ideal for classification (use BCE instead)
    """

    def forward(self, predictions, targets):
        """
        predictions : list of Tensor  (model outputs)
        targets     : list of float/int/Tensor  (ground truth)

        Returns a single scalar Tensor — the mean squared error.
        """
        assert len(predictions) == len(targets), (
            f"predictions and targets must have equal length, "
            f"got {len(predictions)} and {len(targets)}"
        )

        n = len(predictions)
        total = Tensor(0.0)

        for pred, tgt in zip(predictions, targets):
            tgt = _to_tensor(tgt)
            diff = pred - tgt
            total = total + diff ** 2

        # Divide by n to get the mean
        # We use Tensor division so the graph stays intact
        loss = total * Tensor(1.0 / n)
        return loss

    def __repr__(self):
        return "MSELoss()"


# ------------------------------------------------------------------
# Binary Cross-Entropy Loss
# ------------------------------------------------------------------

class BinaryCrossEntropy(Module):
    """
    Binary Cross-Entropy Loss.

    Formula:
        L = -(1/n) * sum( y*log(p) + (1-y)*log(1-p) )

    Where:
        p = predicted probability (must be in (0, 1))
        y = true label (0 or 1)

    Use when:
        - Binary classification (two-class problems)
        - Model outputs probabilities (use sigmoid activation on output)

    Why BCE beats MSE for classification:
        With MSE, a confident wrong prediction (e.g. pred=0.99, target=0)
        produces loss = 0.98. With BCE, the same case produces
        loss = -log(0.01) ≈ 4.6. BCE punishes confidence in wrong
        answers far more aggressively, which trains faster and better.

    Numerical stability:
        log(0) is undefined (-infinity). We clamp predictions to
        [1e-7, 1 - 1e-7] to prevent this. This is standard practice.
    """

    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        """
        predictions : list of Tensor  — values in (0, 1) after sigmoid
        targets     : list of 0/1 values (float or Tensor)

        Returns a single scalar Tensor — the mean BCE loss.
        """
        assert len(predictions) == len(targets), (
            f"predictions and targets must have equal length, "
            f"got {len(predictions)} and {len(targets)}"
        )

        n = len(predictions)
        total = Tensor(0.0)

        for pred, tgt in zip(predictions, targets):
            tgt = _to_tensor(tgt)

            # Clamp prediction to avoid log(0)
            p = max(self.epsilon, min(1.0 - self.epsilon, pred.data))
            p_tensor = Tensor(p)

            # BCE for one sample:
            # -[ y * log(p) + (1-y) * log(1-p) ]
            log_p      = Tensor(math.log(p))
            log_1_mp   = Tensor(math.log(1.0 - p))

            # Build contribution as a Tensor expression so grad flows
            # through tgt and into the graph
            term = tgt * log_p + (Tensor(1.0) - tgt) * log_1_mp
            total = total + (Tensor(0.0) - term)   # negate

        loss = total * Tensor(1.0 / n)
        return loss

    def __repr__(self):
        return f"BinaryCrossEntropy(epsilon={self.epsilon})"


# ------------------------------------------------------------------
# Convenience: compute accuracy
# ------------------------------------------------------------------

def binary_accuracy(predictions, targets, threshold=0.5):
    """
    Compute fraction of correct binary predictions.

    predictions : list of Tensor (raw model outputs)
    targets     : list of 0/1
    threshold   : float, default 0.5

    Returns float in [0, 1].
    This is not differentiable — for evaluation only.
    """
    correct = 0
    for pred, tgt in zip(predictions, targets):
        pred_val = pred.data if isinstance(pred, Tensor) else pred
        tgt_val  = tgt.data  if isinstance(tgt,  Tensor) else tgt
        predicted_class = 1 if pred_val >= threshold else 0
        correct += int(predicted_class == int(tgt_val))
    return correct / len(predictions)

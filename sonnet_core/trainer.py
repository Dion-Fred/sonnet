"""
sonnet/sonnet_core/trainer.py

Sonnet Trainer

Abstracts the training loop so examples stay clean.
Handles:
    - forward pass per sample
    - loss accumulation
    - zero_grad / backward / parameter update
    - logging at configurable intervals
    - loss history for analysis
"""

from sonnet_core.tensor import Tensor


class Trainer:
    """
    Trains a Sonnet model using gradient descent.

    Parameters:
        model        — any Module with .parameters() and .forward()
        loss_fn      — any loss Module with .forward(preds, targets)
        lr           — learning rate (default: 0.01)
        log_every    — print loss every N steps (default: 10)

    Usage:
        trainer = Trainer(model, MSELoss(), lr=0.05)
        history = trainer.train(X, y, steps=200)
    """

    def __init__(self, model, loss_fn, lr=0.01, log_every=10):
        self.model    = model
        self.loss_fn  = loss_fn
        self.lr       = lr
        self.log_every = log_every
        self.history   = []   # loss per step, for analysis

    def train(self, X, y, steps=100):
        """
        Run the training loop.

        X     : list of input vectors (each a list of float)
        y     : list of target values (float or int)
        steps : number of gradient descent steps

        Returns list of loss values (one per step).
        """
        for step in range(steps):

            # ----------------------------------------------------------
            # 1. Forward pass — run model on every training sample
            # ----------------------------------------------------------
            predictions = [self.model(x) for x in X]

            # ----------------------------------------------------------
            # 2. Compute loss over all predictions
            # ----------------------------------------------------------
            loss = self.loss_fn(predictions, y)

            # ----------------------------------------------------------
            # 3. Zero gradients
            #    Must happen before backward() to avoid accumulation
            #    from the previous step.
            # ----------------------------------------------------------
            self.model.zero_grad()

            # ----------------------------------------------------------
            # 4. Backward pass — computes all gradients via autograd
            # ----------------------------------------------------------
            loss.backward()

            # ----------------------------------------------------------
            # 5. Gradient descent parameter update
            #    p = p - lr * dp
            #    We modify .data in-place to preserve Tensor references.
            # ----------------------------------------------------------
            for p in self.model.parameters():
                p.data -= self.lr * p.grad

            # ----------------------------------------------------------
            # 6. Record and log
            # ----------------------------------------------------------
            self.history.append(loss.data)

            if (step + 1) % self.log_every == 0 or step == 0:
                print(f"  step {step+1:>4d}/{steps}  |  loss: {loss.data:.6f}")

        return self.history

    def evaluate(self, X, y, metric_fn=None):
        """
        Run model on X without updating parameters.

        metric_fn : optional callable(predictions, targets) → float
                    e.g. binary_accuracy from loss.py

        Returns (predictions, metric_value or None).
        """
        predictions = [self.model(x) for x in X]
        metric = metric_fn(predictions, y) if metric_fn else None
        return predictions, metric

    def __repr__(self):
        return (f"Trainer(model={self.model.__class__.__name__}, "
                f"loss={self.loss_fn}, lr={self.lr})")

"""
sonnet/examples/xor.py

XOR Training Example

XOR is the canonical proof that a multi-layer network
can learn non-linearly separable patterns.

Truth table:
    [0, 0] → 0
    [0, 1] → 1
    [1, 0] → 1
    [1, 1] → 0

A single neuron (linear classifier) cannot solve this.
A two-layer MLP with nonlinearity can solve it perfectly.

Run from project root:
    python -m examples.xor
"""

import sys
import os
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sonnet_core.nn import MLP
from sonnet_core.loss import MSELoss, binary_accuracy
from sonnet_core.trainer import Trainer


def main():
    random.seed(42)

    # ------------------------------------------------------------------
    # Dataset — all four XOR cases
    # ------------------------------------------------------------------
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    y = [0.0, 1.0, 1.0, 0.0]

    # ------------------------------------------------------------------
    # Model
    # Two hidden neurons is actually enough for XOR, but 4 trains
    # more reliably with random initialization.
    # Architecture: 2 → 4 → 1
    # ------------------------------------------------------------------
    model = MLP(2, [4, 1])

    print("=" * 50)
    print("Sonnet XOR Training")
    print("=" * 50)
    print(f"\nModel: {model}\n")

    # ------------------------------------------------------------------
    # Baseline — predictions before any training
    # ------------------------------------------------------------------
    print("--- Before training ---")
    for xi, yi in zip(X, y):
        pred = model(xi)
        print(f"  input={xi}  target={yi}  pred={pred.data:.4f}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("\n--- Training ---")
    trainer = Trainer(model, MSELoss(), lr=0.1, log_every=25)
    history = trainer.train(X, y, steps=500)

    # ------------------------------------------------------------------
    # Results after training
    # ------------------------------------------------------------------
    print("\n--- After training ---")
    predictions, acc = trainer.evaluate(X, y, metric_fn=binary_accuracy)
    for xi, yi, pred in zip(X, y, predictions):
        label = "✓" if (pred.data >= 0.5) == bool(yi) else "✗"
        print(f"  {label} input={xi}  target={yi}  pred={pred.data:.4f}")

    print(f"\nAccuracy : {acc * 100:.1f}%")
    print(f"Final loss: {history[-1]:.6f}")

    # ------------------------------------------------------------------
    # Loss curve summary
    # ------------------------------------------------------------------
    print("\n--- Loss curve (every 50 steps) ---")
    for i in range(0, len(history), 50):
        bar_len = int(history[i] * 40)
        bar = "█" * bar_len
        print(f"  step {i+1:>4d}: {history[i]:.4f}  {bar}")

    # ------------------------------------------------------------------
    # Decision boundary probe
    # Points near the boundary show the network learned a curved
    # decision boundary — impossible with a linear model.
    # ------------------------------------------------------------------
    print("\n--- Decision boundary probe ---")
    print("  (values > 0.5 classified as 1)\n")
    steps_n = 5
    for row in range(steps_n):
        line = "  "
        for col in range(steps_n):
            xv = col / (steps_n - 1)
            yv = row / (steps_n - 1)
            out = model([xv, yv])
            char = "█" if out.data > 0.5 else "░"
            line += char + " "
        print(line)
    print()


if __name__ == "__main__":
    main()

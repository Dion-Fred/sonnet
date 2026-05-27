"""
sonnet/examples/train_circles.py

Circles Classification — first real dataset training in Sonnet.

Two concentric rings of points:
    Inner ring (r≈0.5) → class 0
    Outer ring (r≈1.0) → class 1

Not linearly separable. Requires a nonlinear decision boundary.
A 2-layer MLP with tanh should reach >95% test accuracy.

Run from project root:
    python -m examples.train_circles
"""

import sys
import os
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sonnet_core.ndtensor import NDTensor
from sonnet_core.linear import Linear, Sequential, Tanh, MSELoss
from sonnet_core.optim import Adam
from sonnet_core.data import make_circles, train_test_split, DataLoader


# ------------------------------------------------------------------
# Accuracy metric
# ------------------------------------------------------------------

def accuracy(model, dataset):
    """
    Compute classification accuracy on a dataset.
    Threshold: output >= 0.5 → class 1, else class 0.
    """
    correct = 0
    for x, y in zip(dataset.X, dataset.y):
        pred = model(NDTensor([x]))   # (1, 2) input
        # output shape: (1, 1) — unwrap to scalar
        pred_val = pred.data[0][0]
        predicted = 1.0 if pred_val >= 0.5 else 0.0
        if predicted == y:
            correct += 1
    return correct / len(dataset)


# ------------------------------------------------------------------
# ASCII decision boundary visualization
# ------------------------------------------------------------------

def show_boundary(model, grid_size=20):
    """
    Print an ASCII map of the model's decision boundary
    over the region [-1.5, 1.5] x [-1.5, 1.5].

    ░ = class 0 predicted (inner, below 0.5)
    █ = class 1 predicted (outer, above 0.5)
    """
    print("\n  Decision boundary (░=class0, █=class1):\n")
    lo, hi = -1.5, 1.5
    for row in range(grid_size):
        yv = hi - (hi - lo) * row / (grid_size - 1)
        line = "  "
        for col in range(grid_size):
            xv = lo + (hi - lo) * col / (grid_size - 1)
            out = model(NDTensor([[xv, yv]]))
            val = out.data[0][0]
            line += "█" if val >= 0.5 else "░"
        print(line)
    print()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    random.seed(42)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = make_circles(n=400, noise=0.05)
    train_set, test_set = train_test_split(dataset, test_ratio=0.2)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    print("=" * 52)
    print("Sonnet — Circles Classification")
    print("=" * 52)
    print(f"\nDataset : {dataset}")
    print(f"Train   : {len(train_set)} samples")
    print(f"Test    : {len(test_set)} samples")
    print(f"Batches : {len(train_loader)} per epoch (batch_size=32)")

    # ------------------------------------------------------------------
    # Model
    # Architecture: 2 → 16 → 16 → 1
    # Two hidden layers gives enough capacity for the circular boundary.
    # ------------------------------------------------------------------
    model = Sequential(
        Linear(2, 16),
        Tanh(),
        Linear(16, 16),
        Tanh(),
        Linear(16, 1),
    )

    loss_fn   = MSELoss()
    optimizer = Adam(model.parameters(), lr=3e-3)

    n_params = sum(
        sum(len(row) if isinstance(row, list) else 1
            for row in p.data)
        if isinstance(p.data, list) else 1
        for p in model.parameters()
    )
    print(f"Model   : {len(model.parameters())} parameter tensors\n")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print("--- Training ---\n")
    epochs = 20

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for X_batch, y_batch in train_loader:
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.data
            n_batches  += 1

        avg_loss = epoch_loss / n_batches

        # Evaluate on test set every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            train_acc = accuracy(model, train_set)
            test_acc  = accuracy(model, test_set)
            bar_len   = int((1.0 - avg_loss) * 20)
            bar       = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  epoch {epoch:>3d}/{epochs}  "
                  f"loss={avg_loss:.4f}  [{bar}]  "
                  f"train={train_acc*100:.1f}%  "
                  f"test={test_acc*100:.1f}%")

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    final_train = accuracy(model, train_set)
    final_test  = accuracy(model, test_set)

    print(f"\n--- Final Results ---")
    print(f"  Train accuracy : {final_train*100:.1f}%")
    print(f"  Test accuracy  : {final_test*100:.1f}%")

    gap = final_train - final_test
    if gap < 0.05:
        print(f"  Generalization : good  (train/test gap = {gap*100:.1f}%)")
    else:
        print(f"  Generalization : some overfitting (gap = {gap*100:.1f}%)")

    # ------------------------------------------------------------------
    # Decision boundary
    # ------------------------------------------------------------------
    show_boundary(model, grid_size=22)

    # ------------------------------------------------------------------
    # Sample predictions
    # ------------------------------------------------------------------
    print("  Sample predictions (inner ring = 0, outer ring = 1):\n")
    test_samples = list(zip(test_set.X[:8], test_set.y[:8]))
    for xv, yv in test_samples:
        pred = model(NDTensor([xv]))
        pv = pred.data[0][0]
        predicted = 1 if pv >= 0.5 else 0
        mark = "✓" if predicted == int(yv) else "✗"
        print(f"  {mark}  x=[{xv[0]:+.3f}, {xv[1]:+.3f}]  "
              f"target={int(yv)}  pred={pv:.3f}")


if __name__ == "__main__":
    main()

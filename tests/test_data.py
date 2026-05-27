"""
sonnet/tests/test_data.py

Tests for Sonnet data pipeline.
Run from project root: python -m tests.test_data
"""

import sys
import os
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sonnet_core.ndtensor import NDTensor
from sonnet_core.data import Dataset, DataLoader, make_circles, make_moons, train_test_split


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, condition, detail=''):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    return condition


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

def test_dataset_len():
    print("\n--- test_dataset_len ---")
    X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    y = [0.0, 1.0, 0.0]
    ds = Dataset(X, y)
    check("length", len(ds) == 3, str(len(ds)))


def test_dataset_getitem():
    print("\n--- test_dataset_getitem ---")
    X = [[1.0, 2.0], [3.0, 4.0]]
    y = [0.0, 1.0]
    ds = Dataset(X, y)
    xi, yi = ds[0]
    check("X[0]", xi == [1.0, 2.0], str(xi))
    check("y[0]", yi == 0.0, str(yi))
    xi1, yi1 = ds[1]
    check("X[1]", xi1 == [3.0, 4.0])
    check("y[1]", yi1 == 1.0)


def test_dataset_slice():
    print("\n--- test_dataset_slice ---")
    X = [[float(i), float(i)] for i in range(10)]
    y = [float(i % 2) for i in range(10)]
    ds = Dataset(X, y)
    Xs, ys = ds[2:5]
    check("slice length X", len(Xs) == 3, str(len(Xs)))
    check("slice length y", len(ys) == 3)


# ------------------------------------------------------------------
# DataLoader
# ------------------------------------------------------------------

def test_dataloader_batch_count():
    print("\n--- test_dataloader_batch_count ---")
    X = [[float(i)] for i in range(100)]
    y = [float(i % 2) for i in range(100)]
    ds = Dataset(X, y)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    # 100 / 32 = 3 full batches + 1 partial = 4
    batches = list(loader)
    check("batch count", len(batches) == 4, str(len(batches)))


def test_dataloader_batch_shapes():
    print("\n--- test_dataloader_batch_shapes ---")
    X = [[float(i), float(i+1)] for i in range(64)]
    y = [float(i % 2) for i in range(64)]
    ds = Dataset(X, y)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    for X_batch, y_batch in loader:
        check("X batch shape", X_batch.shape == (16, 2), str(X_batch.shape))
        check("y batch shape", y_batch.shape == (16, 1), str(y_batch.shape))
        break  # just check first batch


def test_dataloader_covers_all_samples():
    print("\n--- test_dataloader_covers_all_samples ---")
    n = 97   # prime — won't divide evenly
    X = [[float(i)] for i in range(n)]
    y = [float(i % 2) for i in range(n)]
    ds = Dataset(X, y)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    total_seen = 0
    for X_batch, y_batch in loader:
        total_seen += X_batch.shape[0]

    check("all samples seen", total_seen == n, str(total_seen))


def test_dataloader_shuffle_changes_order():
    print("\n--- test_dataloader_shuffle_changes_order ---")
    X = [[float(i)] for i in range(50)]
    y = [float(i % 2) for i in range(50)]
    ds = Dataset(X, y)

    loader = DataLoader(ds, batch_size=50, shuffle=True)
    random.seed(1)
    batch1_X, _ = next(iter(loader))

    random.seed(99)
    batch2_X, _ = next(iter(loader))

    # First elements should differ (different shuffle order)
    check("shuffle produces different order",
          batch1_X.data[0] != batch2_X.data[0],
          f"{batch1_X.data[0]} vs {batch2_X.data[0]}")


def test_dataloader_no_shuffle_is_deterministic():
    print("\n--- test_dataloader_no_shuffle_is_deterministic ---")
    X = [[float(i), float(i)] for i in range(20)]
    y = [float(i % 2) for i in range(20)]
    ds = Dataset(X, y)

    loader = DataLoader(ds, batch_size=20, shuffle=False)
    batch1, _ = next(iter(loader))
    batch2, _ = next(iter(loader))

    check("no-shuffle is deterministic",
          batch1.data == batch2.data)


def test_dataloader_yields_ndtensors():
    print("\n--- test_dataloader_yields_ndtensors ---")
    X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    y = [0.0, 1.0, 0.0, 1.0]
    ds = Dataset(X, y)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    X_batch, y_batch = next(iter(loader))
    check("X is NDTensor", isinstance(X_batch, NDTensor))
    check("y is NDTensor", isinstance(y_batch, NDTensor))


# ------------------------------------------------------------------
# Dataset generators
# ------------------------------------------------------------------

def test_make_circles_size():
    print("\n--- test_make_circles_size ---")
    ds = make_circles(n=200)
    check("total samples", len(ds) == 200, str(len(ds)))
    check("feature dim", len(ds.X[0]) == 2, str(len(ds.X[0])))


def test_make_circles_classes():
    print("\n--- test_make_circles_classes ---")
    ds = make_circles(n=200)
    labels = set(ds.y)
    check("two classes", labels == {0.0, 1.0}, str(labels))
    # Check roughly balanced
    n_class0 = sum(1 for yi in ds.y if yi == 0.0)
    n_class1 = sum(1 for yi in ds.y if yi == 1.0)
    check("balanced classes", n_class0 == n_class1,
          f"class0={n_class0}, class1={n_class1}")


def test_make_circles_radii():
    print("\n--- test_make_circles_radii ---")
    import math
    ds = make_circles(n=400, noise=0.01, seed=0)
    # Class 0 points should cluster near r=0.5
    # Class 1 points should cluster near r=1.0
    r0 = [math.sqrt(x[0]**2 + x[1]**2)
          for x, y in zip(ds.X, ds.y) if y == 0.0]
    r1 = [math.sqrt(x[0]**2 + x[1]**2)
          for x, y in zip(ds.X, ds.y) if y == 1.0]
    mean_r0 = sum(r0) / len(r0)
    mean_r1 = sum(r1) / len(r1)
    check("inner ring radius ≈ 0.5", abs(mean_r0 - 0.5) < 0.05,
          f"mean_r0={mean_r0:.3f}")
    check("outer ring radius ≈ 1.0", abs(mean_r1 - 1.0) < 0.05,
          f"mean_r1={mean_r1:.3f}")


def test_make_moons_size():
    print("\n--- test_make_moons_size ---")
    ds = make_moons(n=300)
    check("total samples", len(ds) == 300, str(len(ds)))
    check("feature dim", len(ds.X[0]) == 2)


# ------------------------------------------------------------------
# Train/test split
# ------------------------------------------------------------------

def test_train_test_split_sizes():
    print("\n--- test_train_test_split_sizes ---")
    ds = make_circles(n=200)
    train, test = train_test_split(ds, test_ratio=0.2)
    check("train size", len(train) == 160, str(len(train)))
    check("test size",  len(test)  == 40,  str(len(test)))
    check("no overlap", len(train) + len(test) == 200)


def test_train_test_split_reproducible():
    print("\n--- test_train_test_split_reproducible ---")
    ds = make_circles(n=100)
    train1, test1 = train_test_split(ds, test_ratio=0.2, seed=7)
    train2, test2 = train_test_split(ds, test_ratio=0.2, seed=7)
    check("same seed same split",
          train1.X == train2.X and test1.X == test2.X)


def test_train_test_split_different_seeds():
    print("\n--- test_train_test_split_different_seeds ---")
    ds = make_circles(n=100)
    train1, _ = train_test_split(ds, test_ratio=0.2, seed=1)
    train2, _ = train_test_split(ds, test_ratio=0.2, seed=2)
    check("different seeds different split",
          train1.X != train2.X)


# ------------------------------------------------------------------
# Integration: DataLoader feeds model
# ------------------------------------------------------------------

def test_dataloader_feeds_model():
    print("\n--- test_dataloader_feeds_model (integration) ---")
    from sonnet_core.linear import Linear, Sequential, Tanh, MSELoss
    from sonnet_core.optim import Adam
    import random as rng
    rng.seed(0)

    ds = make_circles(n=64)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = Sequential(Linear(2, 8), Tanh(), Linear(8, 1))
    loss_fn = MSELoss()
    opt = Adam(model.parameters(), lr=1e-2)

    losses = []
    for X_batch, y_batch in loader:
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        losses.append(loss.data)
        opt.zero_grad()
        loss.backward()
        opt.step()

    check("all batches produced valid loss",
          all(l >= 0 for l in losses),
          f"{len(losses)} batches, losses={[f'{l:.3f}' for l in losses]}")


# ------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_dataset_len,
        test_dataset_getitem,
        test_dataset_slice,
        test_dataloader_batch_count,
        test_dataloader_batch_shapes,
        test_dataloader_covers_all_samples,
        test_dataloader_shuffle_changes_order,
        test_dataloader_no_shuffle_is_deterministic,
        test_dataloader_yields_ndtensors,
        test_make_circles_size,
        test_make_circles_classes,
        test_make_circles_radii,
        test_make_moons_size,
        test_train_test_split_sizes,
        test_train_test_split_reproducible,
        test_train_test_split_different_seeds,
        test_dataloader_feeds_model,
    ]
    for t in tests:
        t()
    print("\nAll tests complete.")

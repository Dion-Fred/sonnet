"""
sonnet/sonnet_core/data.py

Sonnet Data Pipeline

Dataset   — wraps (X, y) pairs, supports indexing and length
DataLoader — iterates over a Dataset in shuffled mini-batches

Design mirrors PyTorch's DataLoader interface so the mental model
transfers directly when you eventually use real frameworks.

Usage:
    dataset = Dataset(X, y)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(10):
        for X_batch, y_batch in loader:
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
"""

import random
from sonnet_core.ndtensor import NDTensor


class Dataset:
    """
    Wraps a list of input vectors X and a list of targets y.

    X : list of lists  — each inner list is one sample's features
    y : list of floats — one target per sample

    Supports:
        len(dataset)          — number of samples
        dataset[i]            — returns (X[i], y[i])
        dataset[i:j]          — returns (X[i:j], y[i:j])
    """

    def __init__(self, X, y):
        assert len(X) == len(y), (
            f"X and y must have same length, got {len(X)} and {len(y)}")
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx]

    def __repr__(self):
        return f"Dataset(n={len(self)}, features={len(self.X[0])})"


class DataLoader:
    """
    Iterates over a Dataset in mini-batches.

    Parameters:
        dataset    — a Dataset object
        batch_size — number of samples per batch (default: 32)
        shuffle    — shuffle order each epoch (default: True)

    Each iteration yields:
        X_batch : NDTensor of shape (batch_size, n_features)
        y_batch : NDTensor of shape (batch_size, 1)

    The last batch may be smaller than batch_size if
    len(dataset) is not divisible by batch_size.

    Why shuffle?
        Neural networks trained on ordered data can overfit
        to sequence patterns that don't exist in new data.
        Shuffling each epoch ensures every batch is a random
        sample from the full dataset.

    Why mini-batches?
        A gradient computed over one sample is noisy.
        Over the full dataset, it's expensive.
        Mini-batches balance noise vs cost: typically 16-256 samples.
        32 is a common default that works well across many tasks.
    """

    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle

    def __iter__(self):
        """
        Each call to iter() produces a fresh sequence of batches.
        If shuffle=True, the index order is randomized first.
        """
        n = len(self.dataset)
        indices = list(range(n))

        if self.shuffle:
            random.shuffle(indices)

        # Yield batches of batch_size until dataset is exhausted
        for start in range(0, n, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]

            # Collect samples for this batch
            X_batch = [self.dataset.X[i] for i in batch_idx]
            y_batch = [self.dataset.y[i] for i in batch_idx]

            # Convert to NDTensor
            # X: (batch, features)
            # y: (batch, 1) — column vector for MSE compatibility
            X_tensor = NDTensor(X_batch)
            y_tensor = NDTensor([[yi] for yi in y_batch])

            yield X_tensor, y_tensor

    def __len__(self):
        """Number of batches per epoch."""
        import math
        return math.ceil(len(self.dataset) / self.batch_size)

    def __repr__(self):
        return (f"DataLoader(n={len(self.dataset)}, "
                f"batch_size={self.batch_size}, "
                f"shuffle={self.shuffle})")


# ------------------------------------------------------------------
# Dataset generators — pure Python, no file I/O needed
# ------------------------------------------------------------------

def make_circles(n=400, noise=0.05, seed=42):
    """
    Generate a two-class concentric circles dataset.

    n     — total number of samples (split evenly between classes)
    noise — standard deviation of Gaussian noise added to coordinates
    seed  — random seed for reproducibility

    Returns Dataset with:
        X : list of [x, y] coordinates
        y : list of 0 (inner circle) or 1 (outer circle)

    This dataset is NOT linearly separable — requires a nonlinear
    decision boundary, which a 2-layer MLP with tanh can learn.
    """
    import math
    random.seed(seed)

    def randn():
        """Box-Muller standard normal sample."""
        while True:
            u1, u2 = random.random(), random.random()
            if u1 > 0:
                return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    X, y = [], []
    n_each = n // 2

    # Inner circle: radius ~0.5
    for _ in range(n_each):
        angle = random.uniform(0, 2 * math.pi)
        r = 0.5 + randn() * noise
        X.append([r * math.cos(angle), r * math.sin(angle)])
        y.append(0.0)

    # Outer circle: radius ~1.0
    for _ in range(n_each):
        angle = random.uniform(0, 2 * math.pi)
        r = 1.0 + randn() * noise
        X.append([r * math.cos(angle), r * math.sin(angle)])
        y.append(1.0)

    return Dataset(X, y)


def make_moons(n=400, noise=0.1, seed=42):
    """
    Generate a two-class interlocking moons dataset.

    Harder than circles — the boundary is more complex.
    Requires a deeper or wider network to solve well.
    """
    import math
    random.seed(seed)

    def randn():
        while True:
            u1, u2 = random.random(), random.random()
            if u1 > 0:
                return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    X, y = [], []
    n_each = n // 2

    # Upper moon
    for i in range(n_each):
        angle = math.pi * i / n_each
        X.append([math.cos(angle) + randn() * noise,
                  math.sin(angle) + randn() * noise])
        y.append(0.0)

    # Lower moon (offset)
    for i in range(n_each):
        angle = math.pi * i / n_each
        X.append([1 - math.cos(angle) + randn() * noise,
                  1 - math.sin(angle) - 0.5 + randn() * noise])
        y.append(1.0)

    return Dataset(X, y)


def train_test_split(dataset, test_ratio=0.2, seed=42):
    """
    Split a Dataset into train and test subsets.

    test_ratio — fraction of data to hold out (default: 0.2 = 20%)

    Returns (train_dataset, test_dataset).

    Why hold out test data?
        Training loss measures how well the model fits the data
        it has seen. Test loss measures generalization — performance
        on data the model has never seen. A model that memorizes
        training data (overfits) will have low training loss but
        high test loss. The gap between them is your signal.
    """
    random.seed(seed)
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)

    split = int(n * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx  = indices[split:]

    train = Dataset(
        [dataset.X[i] for i in train_idx],
        [dataset.y[i] for i in train_idx]
    )
    test = Dataset(
        [dataset.X[i] for i in test_idx],
        [dataset.y[i] for i in test_idx]
    )
    return train, test

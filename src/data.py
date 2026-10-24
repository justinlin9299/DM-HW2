"""Data loading helpers for the preprocessed WM-811K arrays.

Files are produced by `python src/preprocess.py` (see docs/preprocessing.md).

Usage:
    import sys; sys.path.append("src")   # when running from a notebook
    from data import load_split, to_onehot, to_fail_mask

    X, y, failure_type, index = load_split("train")   # X: (N, 64, 64) uint8 in {0, 1, 2}
    X_train_normal = X[y == 0]                          # Tier 2 one-class training set
    X_oh = to_onehot(X)                                 # (N, 3, 64, 64) float32
    X_fail = to_fail_mask(X)                            # (N, 1, 64, 64) float32
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
SIZE = 64


def resize_nearest(m, size=SIZE):
    """Nearest-neighbor resize of a 2D categorical map to (size, size).

    Nearest-neighbor keeps pixel values in {0, 1, 2}; bilinear would create invalid values.
    """
    h, w = m.shape
    rows = (np.arange(size) * h / size).astype(int)
    cols = (np.arange(size) * w / size).astype(int)
    return m[np.ix_(rows, cols)]


def load_split(split, processed_dir=PROCESSED_DIR):
    """Load one split ('train', 'val' or 'test').

    Returns
      X            : (N, 64, 64) uint8, 0 = background, 1 = pass die, 2 = fail die
      y            : (N,) int8, 0 = normal, 1 = anomaly
      failure_type : (N,) str, 9-class label ('none' + 8 defect types)
      index        : (N,) int64, row index in data/LSWMD_clean.pkl and splits/split_v1.csv
    """
    with np.load(Path(processed_dir) / f"{split}.npz") as f:
        return f["X"], f["y"], f["failure_type"], f["index"]


def to_onehot(X):
    """(N, H, W) in {0,1,2} -> (N, 3, H, W) float32 one-hot (channels: background, pass, fail)."""
    return np.stack([X == v for v in (0, 1, 2)], axis=1).astype(np.float32)


def to_fail_mask(X):
    """(N, H, W) in {0,1,2} -> (N, 1, H, W) float32, 1 where the die failed."""
    return (X == 2)[:, None].astype(np.float32)

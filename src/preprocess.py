"""Resize every labeled wafer map to 64x64 and save one .npz per split.

Output: data/processed/{train,val,test}.npz with arrays X, y, failure_type, index
        figures/preprocessing/resize_check.{png,pdf}

Usage:
    python src/preprocess.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import ROOT, SIZE, resize_nearest

SPLITS = ["train", "val", "test"]


def fail_rate(m):
    return (m == 2).sum() / max((m > 0).sum(), 1)


def plot_resize_check(data, X_all, out_path):
    """Original vs resized for one wafer per class (two for Scratch, the thinnest pattern)."""
    picks = []
    for t in ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]:
        rows = data[(data["failureType"] == t) & (data["split"] == "train")]
        picks += list(rows.sample(2 if t == "Scratch" else 1, random_state=1).index)

    fig, axes = plt.subplots(2, len(picks), figsize=(1.6 * len(picks), 3.8))
    for j, idx in enumerate(picks):
        orig = data.at[idx, "waferMap"]
        axes[0, j].imshow(orig, cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
        axes[0, j].set_title(f"{data.at[idx, 'failureType']}\n{orig.shape[0]}×{orig.shape[1]}", fontsize=8)
        axes[1, j].imshow(X_all[idx], cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
        axes[1, j].set_title(f"{SIZE}×{SIZE}", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("original", fontsize=9)
    axes[1, 0].set_ylabel("resized", fontsize=9)
    fig.suptitle("Nearest-neighbor resize check (train split; 0 = background, 1 = pass, 2 = fail)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "LSWMD_clean.pkl")
    parser.add_argument("--split", type=Path, default=ROOT / "splits" / "split_v1.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data} ...")
    df = pd.read_pickle(args.data)
    split = pd.read_csv(args.split, index_col="index")
    data = df.loc[split.index, ["waferMap"]].join(split[["failureType", "label", "split"]])
    del df

    print(f"Resizing {len(data):,} wafer maps to {SIZE}x{SIZE} ...")
    X_all = pd.Series([resize_nearest(m) for m in data["waferMap"]], index=data.index)

    # ---- checks ---------------------------------------------------------------------
    values = np.unique(np.stack(X_all.to_numpy()))
    assert set(values) <= {0, 1, 2}, f"unexpected pixel values {values}"
    fr_orig = data["waferMap"].apply(fail_rate)
    fr_new = X_all.apply(fail_rate)
    fr_err = (fr_orig - fr_new).abs()
    lost = X_all.apply(lambda m: (m == 2).sum() == 0) & data["waferMap"].apply(lambda m: (m == 2).sum() > 0)

    # ---- save one file per split ------------------------------------------------------
    summary = []
    for s in SPLITS:
        part = data[data["split"] == s]
        X = np.stack(X_all.loc[part.index].to_numpy()).astype(np.uint8)
        np.savez_compressed(
            args.out / f"{s}.npz",
            X=X,
            y=part["label"].to_numpy(dtype=np.int8),
            failure_type=part["failureType"].to_numpy(dtype=str),
            index=part.index.to_numpy(dtype=np.int64),
        )
        size_mb = (args.out / f"{s}.npz").stat().st_size / 1e6
        summary.append({"split": s, "N": len(part), "normal": int((part["label"] == 0).sum()),
                        "defect": int((part["label"] == 1).sum()), "X_shape": str(X.shape), "file_MB": round(size_mb, 1)})

    plot_resize_check(data, X_all, ROOT / "figures" / "preprocessing" / "resize_check")

    print("\nSaved")
    print(pd.DataFrame(summary).set_index("split").to_string(), "\n")
    print(f"Pixel values after resize: {values.tolist()}")
    print(f"Fail-rate change |original - resized|: mean={fr_err.mean():.4f}, "
          f"p95={fr_err.quantile(0.95):.4f}, max={fr_err.max():.4f}")
    print("Mean fail-rate change by type:")
    print(fr_err.groupby(data["failureType"]).mean().round(4).to_string())
    print(f"Wafers whose fail dies all vanished after resize: {int(lost.sum())}")
    print(f"Figure: {ROOT / 'figures' / 'preprocessing' / 'resize_check.png'}")


if __name__ == "__main__":
    main()

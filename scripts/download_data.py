"""Download the raw WM-811K pickle from Kaggle into data/LSWMD.pkl.

Requires a Kaggle API token (https://www.kaggle.com/docs/api) and `pip install kagglehub`.
Alternatively, download the zip manually from
https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map and extract LSWMD.pkl into data/.

Usage:
    python scripts/download_data.py
"""
import shutil
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "LSWMD.pkl"


def main():
    if TARGET.exists():
        print(f"Already exists: {TARGET}")
        return
    cache_dir = Path(kagglehub.dataset_download("qingyi/wm811k-wafer-map"))
    src = next(cache_dir.rglob("LSWMD.pkl"))
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying {src} -> {TARGET} (~2 GB) ...")
    shutil.copy2(src, TARGET)
    print("Done.")


if __name__ == "__main__":
    main()

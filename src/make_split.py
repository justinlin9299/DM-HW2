"""Create the shared train/val/test split for WM-811K.

All three tiers MUST use the split file produced here.

Rules (see docs/split.md):
  - Only labeled wafers (failureType != 'unlabeled') are used.
  - Binary label: 'none' -> 0 (normal), any of the 8 defect types -> 1 (anomaly).
  - Grouped by lotName, so no lot appears in more than one split (prevents leakage).
  - Stratified by the 9-class failureType, so every defect type is spread across splits.
  - Ratio train/val/test = 60/20/20, fixed seed.

Usage:
    python src/make_split.py
    python src/make_split.py --data data/LSWMD_clean.pkl --out splits/split_v1.csv --seed 42
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["train", "val", "test"]


def first_fold(y, groups, n_splits, seed):
    """Return (rest_idx, held_idx) positions of the first fold of a StratifiedGroupKFold."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return next(sgkf.split(X=y, y=y, groups=groups))


def make_split(df, seed):
    df = df[df["failureType"] != "unlabeled"].copy()
    df["label"] = (df["failureType"] != "none").astype(int)

    # Step 1: 5 folds -> 1 fold (20%) as test
    rest_pos, test_pos = first_fold(df["failureType"].values, df["lotName"].values, 5, seed)

    # Step 2: remaining 80% -> 4 folds -> 1 fold (20% of total) as val, rest (60%) as train
    rest = df.iloc[rest_pos]
    train_pos, val_pos = first_fold(rest["failureType"].values, rest["lotName"].values, 4, seed)

    df["split"] = ""
    df.iloc[test_pos, df.columns.get_loc("split")] = "test"
    df.loc[rest.index[val_pos], "split"] = "val"
    df.loc[rest.index[train_pos], "split"] = "train"

    out = df[["lotName", "waferIndex", "failureType", "label", "split"]].copy()
    out["waferIndex"] = out["waferIndex"].astype(int)
    out.index.name = "index"
    return out


def check_split(split):
    """Sanity checks; raises AssertionError if the split is invalid."""
    assert (split["split"] != "").all(), "some rows were not assigned"

    lots = {s: set(split.loc[split["split"] == s, "lotName"]) for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            overlap = lots[a] & lots[b]
            assert not overlap, f"{len(overlap)} lots shared between {a} and {b}"

    per_type = pd.crosstab(split["failureType"], split["split"])
    assert (per_type[SPLITS] > 0).all().all(), "some failureType missing from a split"


def summarize(split):
    n = len(split)
    rows = []
    for s in SPLITS:
        part = split[split["split"] == s]
        rows.append({
            "split": s,
            "wafers": len(part),
            "share": f"{len(part) / n:.1%}",
            "lots": part["lotName"].nunique(),
            "normal": int((part["label"] == 0).sum()),
            "defect": int((part["label"] == 1).sum()),
            "defect_rate": f"{part['label'].mean():.2%}",
        })
    overview = pd.DataFrame(rows).set_index("split")

    per_type = pd.crosstab(split["failureType"], split["split"])[SPLITS]
    per_type = per_type.loc[per_type.sum(axis=1).sort_values(ascending=False).index]
    return overview, per_type


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "LSWMD_clean.pkl")
    parser.add_argument("--out", type=Path, default=ROOT / "splits" / "split_v1.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.data} ...")
    df = pd.read_pickle(args.data)

    split = make_split(df, args.seed)
    check_split(split)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(args.out)
    print(f"Saved {len(split):,} rows -> {args.out}\n")

    overview, per_type = summarize(split)
    print("Overview")
    print(overview.to_string(), "\n")
    print("Per failureType")
    print(per_type.to_string(), "\n")
    print("All checks passed: no lot overlap, every failureType present in every split.")


if __name__ == "__main__":
    main()

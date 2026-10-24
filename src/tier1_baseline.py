"""Tier 1 naive baselines for WM-811K anomaly detection.

Methods
  T1a  Predict-all-normal : score = 0 for every wafer, never raises an alarm.
  T1b  Fail-rate threshold: score s(x) = failed dies / total dies.
         T1b-3sigma : tau = mean + 3*std of fail rate over TRAIN normals.
         T1b-bestF1 : tau = F1-maximizing threshold on VAL.

Protocol (see docs/split.md)
  - tau is chosen from train (3sigma) or val (bestF1) only.
  - test is evaluated exactly once.

Usage:
    python src/tier1_baseline.py
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import evaluate, find_best_f1_threshold, per_type_recall

ROOT = Path(__file__).resolve().parents[1]


def fail_rate(m):
    """s(x) = (#dies with value 2) / (#dies with value > 0)."""
    return (m == 2).sum() / (m > 0).sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "LSWMD_clean.pkl")
    parser.add_argument("--split", type=Path, default=ROOT / "splits" / "split_v1.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "tier1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data} ...")
    df = pd.read_pickle(args.data)
    split = pd.read_csv(args.split, index_col="index")
    data = df.loc[split.index, ["waferMap"]].join(split[["failureType", "label", "split"]])
    del df

    parts = {s: data[data["split"] == s] for s in ["train", "val", "test"]}

    # ---- "training": compute scores on train/val and pick tau --------------------------
    t0 = time.perf_counter()
    score_train = parts["train"]["waferMap"].apply(fail_rate)
    score_val = parts["val"]["waferMap"].apply(fail_rate)
    normal_train = score_train[parts["train"]["label"] == 0]
    tau_3sigma = float(normal_train.mean() + 3 * normal_train.std())
    tau_bestf1 = find_best_f1_threshold(parts["val"]["label"], score_val)
    fit_seconds = time.perf_counter() - t0

    # ---- test: score once, timed ------------------------------------------------------
    test = parts["test"]
    t0 = time.perf_counter()
    score_test = test["waferMap"].apply(fail_rate).to_numpy()
    ms_per_sample = (time.perf_counter() - t0) * 1000 / len(test)

    y_test = test["label"].to_numpy()
    types_test = test["failureType"].to_numpy()

    methods = {
        # name: (scores, tau, ms/sample)
        "T1a_all_normal": (np.zeros(len(test)), np.inf, 0.0),
        "T1b_failrate_3sigma": (score_test, tau_3sigma, ms_per_sample),
        "T1b_failrate_bestF1": (score_test, tau_bestf1, ms_per_sample),
    }

    rows, recalls = [], {}
    for name, (scores, tau, ms) in methods.items():
        m = evaluate(y_test, scores, tau)
        m.update({"method": name, "ms_per_sample": ms, "fit_seconds": 0.0 if name == "T1a_all_normal" else fit_seconds})
        rows.append(m)
        recalls[name] = per_type_recall(types_test, scores, tau)["rate"]

    metrics = pd.DataFrame(rows).set_index("method")
    cols = ["pr_auc", "roc_auc", "f1", "precision", "recall", "fpr", "accuracy", "tau",
            "tp", "fp", "fn", "tn", "ms_per_sample", "fit_seconds"]
    metrics = metrics[cols]
    metrics.to_csv(args.out / "metrics.csv", float_format="%.4f")

    counts = pd.Series(types_test).value_counts().rename("n_test")
    per_type = pd.DataFrame(recalls).join(counts)
    order = ["none"] + per_type.drop(index="none").sort_values("T1b_failrate_bestF1").index.tolist()
    per_type = per_type.loc[order]
    per_type.to_csv(args.out / "per_type_recall.csv", float_format="%.4f")

    pd.DataFrame({
        "index": test.index,
        "failureType": types_test,
        "label": y_test,
        "score_failrate": score_test,
        "pred_3sigma": (score_test >= tau_3sigma).astype(int),
        "pred_bestF1": (score_test >= tau_bestf1).astype(int),
    }).to_csv(args.out / "scores_test.csv", index=False, float_format="%.6f")

    pd.set_option("display.width", 200)
    print(f"\ntau_3sigma = {tau_3sigma:.4f} (train normals: mean={normal_train.mean():.4f}, std={normal_train.std():.4f})")
    print(f"tau_bestF1 = {tau_bestf1:.4f} (chosen on val)\n")
    print("Test metrics")
    print(metrics.round(4).to_string(), "\n")
    print("Per-type detection rate on test ('none' row = false alarm rate)")
    print(per_type.round(3).to_string(), "\n")
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()

"""Isolation Forest on handcrafted spatial features (one-class, train normals only).

Features (61 per wafer, computed on the 64x64 processed maps; see docs/iforest.md)
  global    (2)  : fail rate, fail rate after a 3x3 median filter (removes isolated fail dies)
  region   (13)  : fail density in 13 zones, in the spirit of Wu et al. 2015:
                   1 center disc (r < 0.33), 4 middle-ring quadrants (0.33 <= r < 0.75),
                   8 outer-ring sectors (r >= 0.75); r = 1 at the outermost die
  outer    (3)  : max / min / std of the 8 outer-sector densities (rotation-invariant Edge-Loc cue)
  radon    (32)  : Radon transform of the fail mask over 12 angles; mean and std over angles
                   of each projection bin, average-pooled to 16 bins
  geometry (11)  : connected components (8-neighbour) of the median-filtered fail mask:
                   #components, largest component area / #dies, its share of all fails,
                   centroid radius, eccentricity, major / minor axis length, bounding-box fill,
                   mean radius and radius std of its dies, second-largest area / #dies

Score s(x) = -IsolationForest.score_samples(x), higher = more anomalous.
Protocol: fit on train normals only, tau = best-F1 on val, test evaluated once (docs/split.md).

Usage:
    python src/iforest.py                     # 3 seeds -> results/iforest/
    python src/iforest.py --limit 3000        # quick smoke test (do not report)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.ensemble import IsolationForest

from data import SIZE, load_split
from evaluate import evaluate, find_best_f1_threshold, per_type_recall

ROOT = Path(__file__).resolve().parents[1]
ANGLES = np.arange(0, 180, 15)  # 12 Radon angles
RADON_BINS = 16

_yy, _xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)


def polar(die):
    """Per-wafer normalized radius r (1 = outermost die) and angle theta, shape (N, H, W)."""
    n = die.sum((1, 2))
    cy = (die * _yy).sum((1, 2)) / n
    cx = (die * _xx).sum((1, 2)) / n
    dy, dx = _yy - cy[:, None, None], _xx - cx[:, None, None]
    d = np.hypot(dy, dx)
    rmax = np.where(die, d, 0).max((1, 2))
    return d / rmax[:, None, None], np.arctan2(dy, dx)


def region_features(fail, die, r, theta):
    """13 zone densities + 3 outer-ring summaries."""
    sector4 = ((theta + np.pi) / (np.pi / 2)).astype(int) % 4
    sector8 = ((theta + np.pi) / (np.pi / 4)).astype(int) % 8
    zones = [die & (r < 0.33)]
    zones += [die & (r >= 0.33) & (r < 0.75) & (sector4 == k) for k in range(4)]
    zones += [die & (r >= 0.75) & (sector8 == k) for k in range(8)]
    dens = np.stack([(fail & z).sum((1, 2)) / np.maximum(z.sum((1, 2)), 1) for z in zones], 1)
    outer = dens[:, 5:]
    return np.column_stack([dens, outer.max(1), outer.min(1), outer.std(1)])


def radon_features(fail):
    """Mean and std over angles of each Radon projection bin (rotate-and-sum)."""
    f = fail.astype(np.float32)
    proj = np.stack([ndimage.rotate(f, a, axes=(1, 2), reshape=False, order=1).sum(1) for a in ANGLES], 1)
    proj = proj.reshape(len(f), len(ANGLES), RADON_BINS, -1).mean(3) / SIZE  # (N, angles, bins)
    return np.column_stack([proj.mean(1), proj.std(1)])


def geometry_features(fail_f, die, r):
    """Connected-component statistics of the median-filtered fail mask."""
    n_wafer = len(fail_f)
    struct = np.zeros((3, 3, 3), bool)
    struct[1] = True  # 8-connectivity inside a wafer, never across wafers
    lab, n_lab = ndimage.label(fail_f, structure=struct)
    out = np.zeros((n_wafer, 11), np.float32)
    n_die = die.sum((1, 2))
    if n_lab == 0:
        return out
    w, ys, xs = np.nonzero(lab)
    l = lab[w, ys, xs]
    comp = pd.DataFrame({"l": l, "w": w, "y": ys, "x": xs, "r": r[w, ys, xs]})
    comp["yy"], comp["xx"], comp["yx"] = comp.y ** 2, comp.x ** 2, comp.y * comp.x
    st = comp.groupby("l").agg(
        w=("w", "first"), area=("y", "size"), my=("y", "mean"), mx=("x", "mean"),
        myy=("yy", "mean"), mxx=("xx", "mean"), myx=("yx", "mean"), ymin=("y", "min"), ymax=("y", "max"),
        xmin=("x", "min"), xmax=("x", "max"), rm=("r", "mean"), rs=("r", "std")).fillna(0)
    st["vy"], st["vx"] = st.myy - st.my ** 2, st.mxx - st.mx ** 2
    st["cxy"] = st.myx - st.my * st.mx
    st = st.sort_values(["w", "area"], ascending=[True, False])
    n_comp = st.groupby("w").size()
    second = st.groupby("w").nth(1).set_index("w")["area"]
    top = st.groupby("w").head(1).set_index("w")
    # covariance eigenvalues -> axis lengths and eccentricity
    tr, det = top.vx + top.vy, top.vx * top.vy - top.cxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    l1, l2 = np.maximum(tr / 2 + disc, 0), np.maximum(tr / 2 - disc, 0)
    ecc = np.sqrt(1 - l2 / np.where(l1 > 0, l1, 1))
    bbox = (top.ymax - top.ymin + 1) * (top.xmax - top.xmin + 1)
    total_fail = fail_f.sum((1, 2))
    idx = top.index.to_numpy()
    out[idx, 0] = n_comp.loc[idx]
    out[idx, 1] = top.area / n_die[idx]
    out[idx, 2] = top.area / np.maximum(total_fail[idx], 1)
    out[idx, 3] = r[idx, top.my.round().astype(int).clip(0, SIZE - 1), top.mx.round().astype(int).clip(0, SIZE - 1)]
    out[idx, 4] = ecc
    out[idx, 5] = 4 * np.sqrt(l1) / SIZE
    out[idx, 6] = 4 * np.sqrt(l2) / SIZE
    out[idx, 7] = top.area / bbox
    out[idx, 8] = top.rm
    out[idx, 9] = top.rs
    out[second.index.to_numpy(), 10] = second.to_numpy() / n_die[second.index.to_numpy()]
    return out


def extract(X, chunk=5000):
    """(N, 64, 64) wafer maps -> (N, 61) float32 features."""
    feats = []
    for i in range(0, len(X), chunk):
        m = X[i:i + chunk]
        die, fail = m > 0, m == 2
        r, theta = polar(die)
        fail_f = ndimage.median_filter(fail.astype(np.uint8), size=(1, 3, 3)).astype(bool) & die
        n_die = die.sum((1, 2))
        glob = np.column_stack([fail.sum((1, 2)) / n_die, fail_f.sum((1, 2)) / n_die])
        feats.append(np.column_stack([glob, region_features(fail, die, r, theta), radon_features(fail),
                                      geometry_features(fail_f, die, r)]).astype(np.float32))
    return np.concatenate(feats)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "iforest")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None, help="subsample each split (smoke test only)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    splits = {}
    for s in ["train", "val", "test"]:
        X, y, ft, idx = load_split(s)
        if args.limit:
            keep = np.random.default_rng(0).permutation(len(X))[:args.limit]
            X, y, ft, idx = X[keep], y[keep], ft[keep], idx[keep]
        splits[s] = (X, y, ft, idx)

    t0 = time.perf_counter()
    F_train = extract(splits["train"][0][splits["train"][1] == 0])  # one-class: train normals only
    F_val = extract(splits["val"][0])
    feat_seconds = time.perf_counter() - t0
    print(f"features: train normal {F_train.shape}, val {F_val.shape} ({feat_seconds:.0f}s)")

    X_test, y_test, ft_test, idx_test = splits["test"]
    _, y_val, _, _ = splits["val"]
    rows, recalls, cols = [], {}, {"index": idx_test, "failureType": ft_test, "label": y_test}
    for seed in args.seeds:
        t0 = time.perf_counter()
        model = IsolationForest(n_estimators=args.n_estimators, random_state=seed, n_jobs=-1).fit(F_train)
        s_val = -model.score_samples(F_val)
        tau = find_best_f1_threshold(y_val, s_val)
        fit_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()  # test: features + scoring, timed end to end
        s_test = -model.score_samples(extract(X_test))
        ms = (time.perf_counter() - t0) * 1000 / len(X_test)

        m = evaluate(y_test, s_test, tau)
        m.update({"method": f"IF_seed{seed}", "ms_per_sample": ms, "fit_seconds": fit_seconds,
                  "val_pr_auc": evaluate(y_val, s_val, tau)["pr_auc"]})
        rows.append(m)
        recalls[f"seed{seed}"] = per_type_recall(ft_test, s_test, tau)["rate"]
        cols[f"score_seed{seed}"], cols[f"pred_seed{seed}"] = s_test, (s_test >= tau).astype(int)
        print(f"seed {seed}: val PR-AUC {m['val_pr_auc']:.4f}  test PR-AUC {m['pr_auc']:.4f}")

    metrics = pd.DataFrame(rows).set_index("method")
    metrics.loc["IF_mean"] = metrics.mean()
    metrics.loc["IF_std"] = metrics.iloc[:-1].std(ddof=0)
    keep = ["pr_auc", "roc_auc", "f1", "precision", "recall", "fpr", "accuracy", "tau",
            "tp", "fp", "fn", "tn", "ms_per_sample", "fit_seconds", "val_pr_auc"]
    metrics[keep].to_csv(args.out / "metrics.csv", float_format="%.4f")

    per_type = pd.DataFrame(recalls)
    per_type["mean"], per_type["std"] = per_type.mean(axis=1), per_type.std(axis=1, ddof=0)
    per_type = per_type.join(pd.Series(ft_test).value_counts().rename("n_test"))
    per_type = per_type.loc[["none"] + per_type.drop(index="none").sort_values("mean").index.tolist()]
    per_type.index.name = "failureType"
    per_type.to_csv(args.out / "per_type_recall.csv", float_format="%.4f")
    pd.DataFrame(cols).to_csv(args.out / "scores_test.csv", index=False, float_format="%.6f")

    pd.set_option("display.width", 200)
    print("\nTest metrics\n", metrics[["pr_auc", "roc_auc", "f1", "recall", "fpr", "tau"]].round(4).to_string())
    print("\nPer-type detection rate ('none' = false alarm rate)\n", per_type.round(3).to_string())
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()

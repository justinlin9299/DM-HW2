"""Main 3-tier comparison table and significance tests (paper §7, main results).

Reads the outputs of tier1_baseline.py and CNN.py (Tier 2 / Tier 3); trains nothing.

    python src/compare.py

Outputs (results/compare/):
    main_table.csv        one row per method: mean ± std over seeds of every test metric,
                          plus recall at a fixed 5% false alarm rate (threshold-free operating point)
    significance.csv      PR-AUC gap of each pair: Welch t-test over seeds and a paired bootstrap
                          95% CI over test wafers (seed-averaged PR-AUC, 1000 resamples)
    per_type_recall.csv   detection rate per failureType at each method's val-chosen tau (seed mean)
    per_type_auroc.csv    threshold-free: ROC-AUC of each defect type vs all test normals (seed mean)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "compare"

# (label in the paper, folder, metrics.csv row prefix, scores_test.csv score columns prefix)
METHODS = [
    ("T1 Fail-rate (best-F1)", RES / "tier1", "T1b_failrate_bestF1", "score_failrate"),
    ("T2 CNN + synthetic defects", RES / "tier2" / "synth", "T2_cnn_synth_seed", "score_seed"),
    ("Isolation Forest (handcrafted)", RES / "iforest", "IF_seed", "score_seed"),
    ("T3 PatchCore (one-class)", RES / "tier3" / "patchcore", "T3_patchcore_seed", "score_seed"),
    ("T3 ResNet-18 (supervised)", RES / "tier3" / "resnet", "T3_resnet18_seed", "score_seed"),
    ("T3 control: T2 CNN (supervised)", RES / "tier3" / "sup", "T3_sup_cnn_seed", "score_seed"),
]
PAIRS = [  # (a, b): gap = PR-AUC(a) - PR-AUC(b)
    ("T2 CNN + synthetic defects", "T1 Fail-rate (best-F1)"),
    ("Isolation Forest (handcrafted)", "T1 Fail-rate (best-F1)"),
    ("T2 CNN + synthetic defects", "Isolation Forest (handcrafted)"),
    ("T2 CNN + synthetic defects", "T3 PatchCore (one-class)"),
    ("T3 ResNet-18 (supervised)", "T2 CNN + synthetic defects"),
    ("T3 control: T2 CNN (supervised)", "T2 CNN + synthetic defects"),
    ("T3 ResNet-18 (supervised)", "T3 control: T2 CNN (supervised)"),
]
METRICS = ["pr_auc", "roc_auc", "f1", "precision", "recall", "fpr", "ms_per_sample", "fit_seconds"]


def recall_at_fpr(y, s, fpr_target=0.05):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def load(folder, row_prefix, score_prefix):
    """Per-seed metric rows and the (N, n_seeds) test score matrix of one method."""
    m = pd.read_csv(folder / "metrics.csv", index_col=0)
    rows = m[m.index.str.startswith(row_prefix)]
    sc = pd.read_csv(folder / "scores_test.csv")
    cols = [c for c in sc.columns if c.startswith(score_prefix)]
    preds = [c.replace("score_", "pred_") for c in cols]
    if row_prefix.startswith("T1b"):  # Tier 1 is deterministic: one row, its own pred column
        preds = ["pred_bestF1"]
    return rows, sc, cols, preds


def bootstrap_gap(y, sa, sb, n_boot=1000, seed=0):
    """Paired bootstrap over test wafers of mean-over-seeds PR-AUC(a) - PR-AUC(b)."""
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        b = rng.integers(0, len(y), len(y))
        gaps[i] = (np.mean([average_precision_score(y[b], s[b]) for s in sa.T])
                   - np.mean([average_precision_score(y[b], s[b]) for s in sb.T]))
    return np.percentile(gaps, [2.5, 97.5])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = [(name, *load(f, r, s)) for name, f, r, s in METHODS if (f / "metrics.csv").exists()]
    missing = [name for name, f, _, _ in METHODS if not (f / "metrics.csv").exists()]
    if missing:
        print("not found, skipped:", missing)

    y = loaded[0][2]["label"].to_numpy()
    ft = loaded[0][2]["failureType"].to_numpy()
    table, scores, per_type, auroc = [], {}, {}, {}
    for name, rows, sc, cols, preds in loaded:
        assert (sc["label"].to_numpy() == y).all(), "all tiers must be scored on the same test set"
        S = sc[cols].to_numpy()
        scores[name] = S
        r = {"method": name, "n_seeds": len(rows)}
        for k in METRICS:
            r[k] = rows[k].mean()
            r[k + "_std"] = rows[k].std(ddof=0)
        rec5 = [recall_at_fpr(y, s) for s in S.T]
        r["recall_at_5fpr"], r["recall_at_5fpr_std"] = np.mean(rec5), np.std(rec5)
        table.append(r)
        P = sc[preds].to_numpy()
        per_type[name] = pd.DataFrame(P).groupby(ft).mean().mean(axis=1)
        auroc[name] = pd.Series({t: np.mean([roc_auc_score(y[(ft == t) | (ft == "none")] > 0,
                                                           s[(ft == t) | (ft == "none")]) for s in S.T])
                                 for t in sorted(set(ft) - {"none"})})
    table = pd.DataFrame(table).set_index("method")
    table.to_csv(OUT / "main_table.csv", float_format="%.4f")

    names = [n for n, *_ in loaded]
    sig = []
    for na, nb in PAIRS:
        if na not in names or nb not in names:
            continue
        pa = [average_precision_score(y, s) for s in scores[na].T]
        pb = [average_precision_score(y, s) for s in scores[nb].T]
        p = stats.ttest_ind(pa, pb, equal_var=False).pvalue if min(len(pa), len(pb)) > 1 else np.nan
        lo, hi = bootstrap_gap(y, scores[na], scores[nb])
        sig.append({"a": na, "b": nb, "gap_pr_auc": np.mean(pa) - np.mean(pb), "welch_p": p,
                    "boot_ci_lo": lo, "boot_ci_hi": hi, "ci_excludes_0": not lo <= 0 <= hi})
        print(f"{na} - {nb}: {sig[-1]['gap_pr_auc']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  Welch p = {p:.4f}")
    pd.DataFrame(sig).to_csv(OUT / "significance.csv", index=False, float_format="%.4f")

    counts = pd.Series(ft).value_counts().rename("n_test")
    order = ["none"] + sorted(set(ft) - {"none"}, key=lambda t: per_type[METHODS[1][0]][t] if METHODS[1][0] in per_type else 0)
    pt = pd.DataFrame(per_type).join(counts).loc[order]
    pt.to_csv(OUT / "per_type_recall.csv", float_format="%.4f")
    au = pd.DataFrame(auroc).join(counts).loc[order[1:]]
    au.to_csv(OUT / "per_type_auroc.csv", float_format="%.4f")

    pd.set_option("display.width", 250)
    show = ["pr_auc", "pr_auc_std", "roc_auc", "f1", "recall", "fpr", "recall_at_5fpr", "ms_per_sample"]
    print("\nMain results (test, mean over seeds)\n", table[show].round(4).to_string())
    print("\nPer-type detection rate at tau\n", pt.round(3).to_string())
    print("\nPer-type ROC-AUC vs normals\n", au.round(3).to_string())
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()

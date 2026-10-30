"""Controlled ablations of the Tier 2 method (paper §7, ablation). One factor at a time.

The retrained ablations come from scripts/run_tier3_ablation.sh (results/ablation/<setting>/synth).
This script adds the threshold-strategy ablation (no retraining, uses the saved Tier 2 models)
and summarizes everything against the baseline = Tier 2 default retrained on the same machine.

    python src/ablation.py

Outputs (results/ablation/):
    summary.csv           every retrained setting: PR-AUC / ROC-AUC / F1 mean ± std, delta vs
                          baseline, Welch p over seeds, and whether |delta| > the larger seed std
    leave_one_out.csv     A2: ROC-AUC of each real defect type vs normals when that type is NOT
                          synthesized, against the baseline (threshold-free, so tau plays no role)
    threshold.csv         A4: test precision / recall / F1 / FPR of the Tier 2 models under
                          different val-only tau rules
    figures/ablation/{contamination,leave_one_out}.{png,pdf}
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

from CNN import DEFECT_TYPES, WaferCNN, score_all
from data import load_split
from evaluate import evaluate, find_best_f1_threshold

ROOT = Path(__file__).resolve().parents[1]
ABL = ROOT / "results" / "ablation"
FIG = ROOT / "figures" / "ablation"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (baseline, variant)

SETTINGS = (["baseline", "binary", "contam_0.01", "contam_0.05", "contam_0.10"]
            + [f"drop_{t}" for t in DEFECT_TYPES])


def seed_rows(setting):
    m = pd.read_csv(ABL / setting / "synth" / "metrics.csv", index_col=0)
    return m[m.index.str.contains("_seed")]


def type_auroc(setting):
    """(n_types, n_seeds) ROC-AUC of each defect type vs all test normals."""
    sc = pd.read_csv(ABL / setting / "synth" / "scores_test.csv")
    normal = sc["failureType"] == "none"
    cols = [c for c in sc.columns if c.startswith("score_seed")]
    return pd.DataFrame({c: {t: roc_auc_score(sc.loc[normal | (sc["failureType"] == t), "label"],
                                              sc.loc[normal | (sc["failureType"] == t), c])
                             for t in DEFECT_TYPES} for c in cols})


def summarize():
    done = [s for s in SETTINGS if (ABL / s / "synth" / "metrics.csv").exists()]
    if "baseline" not in done:
        raise SystemExit("results/ablation/baseline not found: run scripts/run_tier3_ablation.sh first")
    base = seed_rows("baseline")
    rows = []
    for s in done:
        r = seed_rows(s)
        row = {"setting": s, "n_seeds": len(r)}
        for k in ("pr_auc", "roc_auc", "f1", "recall", "fpr"):
            row[k], row[k + "_std"] = r[k].mean(), r[k].std(ddof=0)
        row["delta_pr_auc"] = row["pr_auc"] - base["pr_auc"].mean()
        row["welch_p"] = (stats.ttest_ind(r["pr_auc"], base["pr_auc"], equal_var=False).pvalue
                          if s != "baseline" else np.nan)
        row["exceeds_seed_std"] = abs(row["delta_pr_auc"]) > max(row["pr_auc_std"], base["pr_auc"].std(ddof=0))
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("setting")
    summary.to_csv(ABL / "summary.csv", float_format="%.4f")
    print("Retrained ablations (test, mean ± std over seeds)\n",
          summary[["pr_auc", "pr_auc_std", "delta_pr_auc", "welch_p", "exceeds_seed_std",
                   "roc_auc", "f1", "recall", "fpr"]].round(4).to_string(), "\n")

    # A2 leave-one-type-out: does detecting real type T depend on synthesizing T?
    b_au = type_auroc("baseline")
    loo = []
    for t in DEFECT_TYPES:
        if f"drop_{t}" not in done:
            continue
        d_au = type_auroc(f"drop_{t}")
        others = [o for o in DEFECT_TYPES if o != t]
        loo.append({"dropped": t,
                    "auroc_base": b_au.loc[t].mean(), "auroc_base_std": b_au.loc[t].std(ddof=0),
                    "auroc_dropped": d_au.loc[t].mean(), "auroc_dropped_std": d_au.loc[t].std(ddof=0),
                    "delta": d_au.loc[t].mean() - b_au.loc[t].mean(),
                    "others_delta": d_au.loc[others].mean().mean() - b_au.loc[others].mean().mean(),
                    "pr_auc_delta": summary.loc[f"drop_{t}", "delta_pr_auc"]})
    if loo:
        loo = pd.DataFrame(loo).set_index("dropped")
        loo.to_csv(ABL / "leave_one_out.csv", float_format="%.4f")
        print("Leave-one-type-out: ROC-AUC of the dropped type vs normals\n", loo.round(4).to_string(), "\n")
    return summary, loo


def threshold_ablation(device):
    """A4: same Tier 2 models and scores, only the val-only rule that picks tau changes."""
    X_val, y_val, _, _ = load_split("val")
    X_test, y_test, _, _ = load_split("test")
    Xv, Xt = torch.from_numpy(X_val).to(device), torch.from_numpy(X_test).to(device)
    args = argparse.Namespace(eval_batch=1024)
    rules = {
        "best F1 on val (default)": lambda s, y: find_best_f1_threshold(y, s),
        "val normals mean + 3 std": lambda s, y: s[y == 0].mean() + 3 * s[y == 0].std(),
        "FPR budget 5% (val normals q95)": lambda s, y: np.quantile(s[y == 0], 0.95),
        "FPR budget 1% (val normals q99)": lambda s, y: np.quantile(s[y == 0], 0.99),
        "review budget 10% of wafers": lambda s, y: np.quantile(s, 0.90),
    }
    res = {r: [] for r in rules}
    for path in sorted((ROOT / "results" / "tier2" / "synth").glob("model_seed*.pt")):
        model = WaferCNN().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        sv, st = score_all(model, Xv, args), score_all(model, Xt, args)
        for r, rule in rules.items():
            res[r].append(evaluate(y_test, st, rule(sv, y_val)))
    rows = []
    for r, ms in res.items():
        df = pd.DataFrame(ms)
        row = {"rule": r}
        for k in ("precision", "recall", "f1", "fpr"):
            row[k], row[k + "_std"] = df[k].mean(), df[k].std(ddof=0)
        row["alerts_per_1000"] = ((df["tp"] + df["fp"]) / len(y_test) * 1000).mean()
        rows.append(row)
    out = pd.DataFrame(rows).set_index("rule")
    out.to_csv(ABL / "threshold.csv", float_format="%.4f")
    print("Threshold strategy (Tier 2 models, test)\n", out.round(4).to_string(), "\n")
    return out


def plot(summary, loo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    cont = [s for s in ["baseline", "contam_0.01", "contam_0.05", "contam_0.10"] if s in summary.index]
    if len(cont) > 1:
        x = [0 if s == "baseline" else float(s.split("_")[1]) * 100 for s in cont]
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        ax.errorbar(x, summary.loc[cont, "pr_auc"], yerr=summary.loc[cont, "pr_auc_std"], color=BLUE,
                    lw=2, marker="o", ms=6, capsize=3)
        ax.set_xlabel("Unlabeled real defects in training pool (%)")
        ax.set_ylabel("Test PR-AUC")
        ax.set_xticks(x)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"contamination.{ext}", dpi=200)
        plt.close(fig)

    if len(loo):
        fig, ax = plt.subplots(figsize=(6.8, 2.5))
        i = np.arange(len(loo))
        w = 0.38
        ax.bar(i - w / 2 - 0.01, loo["auroc_base"], w, yerr=loo["auroc_base_std"], color=BLUE,
               label="all 8 types synthesized (baseline)", capsize=2)
        ax.bar(i + w / 2 + 0.01, loo["auroc_dropped"], w, yerr=loo["auroc_dropped_std"], color=ORANGE,
               label="this type NOT synthesized", capsize=2)
        ax.set_xticks(i, loo.index)
        ax.set_ylabel("ROC-AUC vs normals")
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"leave_one_out.{ext}", dpi=200)
        plt.close(fig)
    print(f"figures saved to {FIG}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary, loo = summarize()
    threshold_ablation(device)
    plot(summary, loo)


if __name__ == "__main__":
    main()

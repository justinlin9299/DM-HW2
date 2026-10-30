"""Tier 3 one-class reference: PatchCore (Roth et al., CVPR 2022).

Same paradigm as Tier 2: only train wafers with y == 0 are used, no defect label ever enters
fitting. VAL labels pick tau (best F1); TEST is scored once.

Pipeline (follows the paper; deviations marked *):
  1. Input: 64x64 one-hot map (background / pass / fail as 3 channels), ImageNet-normalized and
     upsampled (nearest) to --size (default 128; *the paper uses 224 for natural images).
  2. Frozen ImageNet WideResNet-50-2. Features of layer2 and layer3, each averaged over a 3x3
     neighbourhood, layer3 upsampled to layer2's grid and concatenated -> one 1536-d vector per
     patch (16x16 = 256 patches per wafer at size 128).
  3. Memory bank: the patch features of ALL train normals (87,970 x 256 = 22.5M patches),
     reduced by greedy coreset subsampling to --coreset of them. Coreset distances are computed
     on a random 128-d projection (Johnson-Lindenstrauss, as in the paper), the bank keeps the
     full 1536-d features.
  4. Score s(x) = max over the wafer's patches of the distance to the nearest bank patch.
     (*The paper's optional re-weighting of this max is not used.)

The seed sets the random projection and the first coreset element, so 3 seeds give mean ± std.

    python src/patchcore.py                                 # 3 seeds -> results/tier3/patchcore/
    python src/patchcore.py --limit 3000 --seeds 0          # smoke test (not reportable)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from CNN import onehot, save_results, set_seed
from data import load_split
from evaluate import evaluate, find_best_f1_threshold, per_type_recall
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
NAME = "T3_patchcore"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Extractor(torch.nn.Module):
    """Frozen WideResNet-50-2 -> locally aware patch features (B, P, 1536)."""

    def __init__(self, size):
        super().__init__()
        from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
        net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.size = size
        self.register_buffer("mean", MEAN)
        self.register_buffer("std", STD)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate((onehot(x) - self.mean) / self.std, size=self.size, mode="nearest")
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)
        f3 = F.interpolate(F.avg_pool2d(f3, 3, stride=1, padding=1), size=f2.shape[-2:], mode="bilinear",
                           align_corners=False)
        f = torch.cat([f2, f3], 1)  # (B, 1536, h, w)
        return f.flatten(2).transpose(1, 2)  # (B, h*w, 1536)


def batches(X, bs):
    for i in range(0, len(X), bs):
        yield i, X[i:i + bs]


@torch.no_grad()
def greedy_coreset(P, n, seed):
    """Indices of n rows of P chosen by greedy k-center (farthest point) selection."""
    g = torch.Generator(device=P.device).manual_seed(seed)
    sq = torch.cat([(c.float() ** 2).sum(1) for c in P.split(1 << 20)])  # chunked: P is ~6 GB
    idx = torch.empty(n, dtype=torch.long, device=P.device)
    idx[0] = torch.randint(len(P), (1,), generator=g, device=P.device)
    min_d = torch.full((len(P),), float("inf"), device=P.device)
    for i in range(1, n):
        c = P[idx[i - 1]]
        d = sq - 2 * (P @ c).float() + sq[idx[i - 1]]
        torch.minimum(min_d, d, out=min_d)
        idx[i] = torch.argmax(min_d)
    return idx


def fit(ext, X_tr, args, seed):
    """Build the memory bank (M, 1536) from train normals."""
    dev = X_tr.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        n_patch, dim = ext(X_tr[:1]).shape[1:]
    proj = torch.randn(dim, args.proj_dim, generator=gen, device=dev).half() / args.proj_dim ** 0.5

    # pass 1: projected features of every train-normal patch, for coreset selection
    P = torch.empty(len(X_tr) * n_patch, args.proj_dim, dtype=torch.float16, device=dev)
    for i, xb in batches(X_tr, args.batch):
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            f = ext(xb).reshape(-1, dim).half()
        P[i * n_patch:(i + len(xb)) * n_patch] = f @ proj
    n_bank = max(1, int(round(args.coreset * len(P))))
    print(f"  {len(P):,} patches -> coreset of {n_bank:,}")
    sel = torch.sort(greedy_coreset(P, n_bank, seed)).values
    del P

    # pass 2: full 1536-d features of the selected patches only
    bank = torch.empty(n_bank, dim, dtype=torch.float16, device=dev)
    img = sel // n_patch
    for i, xb in batches(X_tr, args.batch):
        m = (img >= i) & (img < i + len(xb))
        if m.any():
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                f = ext(xb).reshape(-1, dim).half()
            bank[m] = f[sel[m] - i * n_patch]
    return bank


@torch.no_grad()
def score(ext, bank, X, args):
    """s(x) = max over patches of the nearest-neighbour distance to the bank."""
    dev = X.device
    bank_sq = (bank.float() ** 2).sum(1)
    out = []
    for _, xb in batches(X, args.score_batch):
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            f = ext(xb).half()
        B, n_patch, dim = f.shape
        q = f.reshape(-1, dim)
        d = ((q.float() ** 2).sum(1, keepdim=True) - 2 * (q @ bank.T).float() + bank_sq).amin(1)
        out.append(d.clamp(min=0).sqrt().reshape(B, n_patch).amax(1).cpu())
    return torch.cat(out).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--size", type=int, default=128, help="backbone input size (nearest upsample of 64x64)")
    parser.add_argument("--coreset", type=float, default=0.002, help="share of train-normal patches kept in the bank")
    parser.add_argument("--proj-dim", type=int, default=128, help="random projection dim for coreset selection")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--score-batch", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="DEBUG ONLY: subsample every split to N wafers")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "tier3")
    args = parser.parse_args()

    out = args.out / ("patchcore" + ("_debug" if args.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{NAME} on {device}")

    X_train, y_train, _, _ = load_split("train")
    X_val, y_val, _, _ = load_split("val")
    X_test, y_test, ft_test, idx_test = load_split("test")
    X_train_normal = X_train[y_train == 0]  # one-class: train normals only
    del X_train
    if args.limit:
        print(f"WARNING: --limit {args.limit} is for debugging; results go to {out} and are not reportable")
        rng = np.random.default_rng(0)
        X_train_normal = X_train_normal[rng.choice(len(X_train_normal), min(args.limit, len(X_train_normal)), replace=False)]
        keep = rng.choice(len(X_val), min(args.limit, len(X_val)), replace=False)
        X_val, y_val = X_val[keep], y_val[keep]
        keep = rng.choice(len(X_test), min(args.limit, len(X_test)), replace=False)
        X_test, y_test, ft_test, idx_test = X_test[keep], y_test[keep], ft_test[keep], idx_test[keep]
    print(f"train normals {len(X_train_normal):,} | val {len(X_val):,} | test {len(X_test):,}")

    ext = Extractor(args.size).to(device)
    X_tr_t = torch.from_numpy(X_train_normal).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)

    rows, recalls = [], {}
    scores_df = pd.DataFrame({"index": idx_test, "failureType": ft_test, "label": y_test})
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        set_seed(seed)
        t0 = time.perf_counter()
        bank = fit(ext, X_tr_t, args, seed)
        fit_seconds = time.perf_counter() - t0

        scores_val = score(ext, bank, X_val_t, args)
        tau = find_best_f1_threshold(y_val, scores_val)  # tau on VAL only
        t0 = time.perf_counter()
        scores_test = score(ext, bank, X_test_t, args)  # TEST scored once
        ms_per_sample = (time.perf_counter() - t0) * 1000 / len(X_test)

        m = evaluate(y_test, scores_test, tau)
        m.update({"method": f"{NAME}_seed{seed}", "seed": seed, "best_epoch": 0,
                  "val_pr_auc": average_precision_score(y_val, scores_val),
                  "ms_per_sample": ms_per_sample, "fit_seconds": fit_seconds})
        rows.append(m)
        recalls[f"seed{seed}"] = per_type_recall(ft_test, scores_test, tau)["rate"]
        scores_df[f"score_seed{seed}"] = scores_test
        scores_df[f"pred_seed{seed}"] = (scores_test >= tau).astype(int)
        print(f"  fit {fit_seconds:.0f}s  val PR-AUC {m['val_pr_auc']:.4f}  test PR-AUC {m['pr_auc']:.4f}"
              f"  ROC-AUC {m['roc_auc']:.4f}  F1 {m['f1']:.4f}  recall {m['recall']:.4f}  FPR {m['fpr']:.4f}")
        del bank
        torch.cuda.empty_cache()

    save_results(rows, recalls, scores_df, NAME, out, ft_test)


if __name__ == "__main__":
    main()

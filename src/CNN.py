"""Tier 2 primary method: CNN anomaly detector trained on train normals only.

Adapted from the reference notebook `Python code.ipynb` (Keras, supervised 9-class CNN on
WM-811K). What we keep from it:
  - the CNN of its final "Model 6": Conv3x3 (He init) -> BatchNorm -> ReLU -> MaxPool 2,
    GlobalMaxPool, softmax over 9 classes (none + 8 defect types), Adam, early stopping on val
  - hand-crafted defect synthesis (it drew Donut / Loc / Near-full / Random / Scratch on a
    defect-free template wafer)
  - flip / 90-degree rotation augmentation

What changes to satisfy the Tier 2 protocol (docs/split.md §5) and other improvements:
  - Training uses ONLY train wafers with y == 0. No real defect label is ever used for training.
    All 8 defect classes are synthesized on the fly on top of real normal wafers (the reference
    used one fixed template and only 5 types), so every batch sees new, realistic backgrounds.
  - Anomaly score s(x) = log P(any defect) - log P(none) = logsumexp(logits[1:]) - logits[0];
    higher = more anomalous, as required by evaluate.py.
  - VAL (real labels) is used only for early stopping (PR-AUC) and for tau. TEST is scored once.
  - One extra conv block (128) because our input is 64x64 instead of 53x52.
  - Augmentation, synthesis and one-hot are done on GPU per batch (no PNG folders / JSON dumps).
  - AdamW + cosine LR, mixed precision, several seeds (mean ± std), optional 8-view TTA.

Methods
  synth (default)  T2_cnn_synth : 9-class CNN, none vs 8 synthetic defect types
  ae               T2_conv_ae   : convolutional autoencoder (ablation, pure reconstruction).
                                  Score = per-pixel CE inside the wafer, avg-pooled, top-k mean,
                                  so clustered errors (patterns) outweigh scattered random fails.

Colab (Runtime -> Change runtime type -> GPU):
    !git clone https://github.com/justinlin9299/DM-HW2.git
    %cd DM-HW2
    !python src/CNN.py                       # main Tier 2 result, 3 seeds
    !python src/CNN.py --method ae           # ablation
    !python src/CNN.py --limit 3000 --epochs 1 --seeds 0   # 1-minute smoke test

Outputs (results/tier2/<method>/), same layout as results/tier1/:
    metrics.csv          one row per seed + mean + std
    per_type_recall.csv  detection rate per failureType ('none' row = false alarm rate)
    scores_test.csv      index, failureType, label, score_seed*, pred_seed*  (§8 error analysis)
    history_seed*.csv    train loss / val PR-AUC per epoch (training curves)
    model_seed*.pt       best weights (by val PR-AUC)
figures/tier2/synthetic_examples.{png,pdf}: what the synthetic training defects look like.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from data import load_split
from evaluate import evaluate, find_best_f1_threshold, per_type_recall

ROOT = Path(__file__).resolve().parents[1]
DEFECT_TYPES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]
METHOD_NAMES = {"synth": "T2_cnn_synth", "ae": "T2_conv_ae"}


# ---- synthetic defects ---------------------------------------------------------------------

def _u(lo, hi, n, device):
    """(n, 1, 1) uniform random tensor, one value per wafer."""
    return torch.empty(n, 1, 1, device=device).uniform_(lo, hi)


def wafer_geometry(x):
    """Polar coordinates of every pixel relative to each wafer's own center.

    x: (B, H, W) uint8. r is normalized so the farthest die is at r = 1 (same as eda.md §2).
    """
    _, H, W = x.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device, dtype=torch.float32),
                            torch.arange(W, device=x.device, dtype=torch.float32), indexing="ij")
    yy, xx = yy[None], xx[None]
    m = (x > 0).float()
    n = m.sum((1, 2), keepdim=True).clamp(min=1)
    cy = (m * yy).sum((1, 2), keepdim=True) / n
    cx = (m * xx).sum((1, 2), keepdim=True) / n
    d = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    R = (d * m).amax((1, 2), keepdim=True).clamp(min=1)
    return {"r": d / R, "theta": torch.atan2(yy - cy, xx - cx), "cy": cy, "cx": cx, "R": R,
            "yy": yy, "xx": xx}


def _seg_dist(g, y0, x0, y1, x1):
    """Distance from every pixel to the segment (y0, x0)-(y1, x1)."""
    vy, vx = y1 - y0, x1 - x0
    t = (((g["yy"] - y0) * vy + (g["xx"] - x0) * vx) / (vy ** 2 + vx ** 2).clamp(min=1e-6)).clamp(0, 1)
    return torch.sqrt((g["yy"] - y0 - t * vy) ** 2 + (g["xx"] - x0 - t * vx) ** 2)


# Each generator returns P (n, H, W): probability that a die turns into a fail die.
# Shapes follow eda.md Fig 3 (mean fail map per type).

def _center(g, n, dev):
    return _u(0.6, 0.95, n, dev) * (g["r"] < _u(0.15, 0.45, n, dev))


def _donut(g, n, dev):
    r1, w = _u(0.15, 0.4, n, dev), _u(0.12, 0.3, n, dev)
    return _u(0.5, 0.9, n, dev) * ((g["r"] > r1) & (g["r"] < r1 + w))


def _edge_loc(g, n, dev):
    th0, half = _u(-math.pi, math.pi, n, dev), _u(math.pi / 12, math.pi / 5, n, dev)
    dth = torch.remainder(g["theta"] - th0 + math.pi, 2 * math.pi) - math.pi
    return _u(0.7, 0.95, n, dev) * ((g["r"] > 1 - _u(0.15, 0.35, n, dev)) & (dth.abs() < half))


def _edge_ring(g, n, dev):
    return _u(0.65, 0.95, n, dev) * (g["r"] > 1 - _u(0.08, 0.2, n, dev))


def _loc(g, n, dev):
    # an elliptical blob away from the edge (rc + size <= 0.87), otherwise it would be Edge-Loc
    a, rc = _u(-math.pi, math.pi, n, dev), _u(0.0, 0.65, n, dev)
    by, bx = g["cy"] + rc * g["R"] * torch.sin(a), g["cx"] + rc * g["R"] * torch.cos(a)
    sy, sx = _u(0.08, 0.22, n, dev) * g["R"], _u(0.08, 0.22, n, dev) * g["R"]
    inside = ((g["yy"] - by) / sy) ** 2 + ((g["xx"] - bx) / sx) ** 2 < 1
    return _u(0.6, 0.95, n, dev) * inside


def _random(g, n, dev):
    return _u(0.2, 0.5, n, dev) * torch.ones_like(g["r"])


def _near_full(g, n, dev):
    return _u(0.8, 0.98, n, dev) * torch.ones_like(g["r"])


def _scratch(g, n, dev):
    # thin polyline of two segments with a small bend, starting anywhere on the wafer
    a, rs = _u(-math.pi, math.pi, n, dev), _u(0.1, 0.9, n, dev)
    y0, x0 = g["cy"] + rs * g["R"] * torch.sin(a), g["cx"] + rs * g["R"] * torch.cos(a)
    phi = _u(-math.pi, math.pi, n, dev)
    l1, l2 = _u(0.15, 0.5, n, dev) * g["R"], _u(0.0, 0.4, n, dev) * g["R"]
    y1, x1 = y0 + l1 * torch.sin(phi), x0 + l1 * torch.cos(phi)
    phi2 = phi + _u(-0.5, 0.5, n, dev)
    y2, x2 = y1 + l2 * torch.sin(phi2), x1 + l2 * torch.cos(phi2)
    dist = torch.minimum(_seg_dist(g, y0, x0, y1, x1), _seg_dist(g, y1, x1, y2, x2))
    return _u(0.85, 1.0, n, dev) * (dist < _u(0.8, 1.6, n, dev))


GENERATORS = [_center, _donut, _edge_loc, _edge_ring, _loc, _near_full, _random, _scratch]


def synthesize(x, types):
    """Draw a defect of class types[i] (1..8 = DEFECT_TYPES, 0 = leave normal) on wafer x[i].

    x: (B, H, W) uint8 normal wafers. Existing random fail dies are kept, so the synthetic
    pattern sits on top of a realistic yield-loss background.
    """
    g = wafer_geometry(x)
    g["r"] = g["r"] + 0.04 * torch.randn_like(g["r"])  # ragged, non-perfect boundaries
    p = torch.zeros_like(g["r"])
    for t, gen in enumerate(GENERATORS, start=1):
        idx = types == t
        if idx.any():
            sub = {k: v if k in ("yy", "xx") else v[idx] for k, v in g.items()}
            p[idx] = gen(sub, int(idx.sum()), x.device)
    fail = (torch.rand_like(p) < p) & (x > 0)
    return torch.where(fail, torch.full_like(x, 2), x)


# ---- augmentation / representation ---------------------------------------------------------

def onehot(x):
    """(B, H, W) uint8 -> (B, 3, H, W) float, same channel order as data.to_onehot."""
    return F.one_hot(x.long(), 3).permute(0, 3, 1, 2).float()


def random_dihedral(x):
    """Random horizontal flip + 0/90/180/270 rotation per wafer (reference: flip + rot 90/270)."""
    flip = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(flip[:, None, None], x.flip(-1), x)
    k = torch.randint(0, 4, (len(x),), device=x.device)
    out = x.clone()
    for r in (1, 2, 3):
        idx = k == r
        out[idx] = torch.rot90(x[idx], r, dims=(-2, -1))
    return out


def dihedral_views(x):
    """All 8 flips/rotations of a batch, for test-time augmentation."""
    for f in (x, x.flip(-1)):
        for k in range(4):
            yield torch.rot90(f, k, dims=(-2, -1))


# ---- models ---------------------------------------------------------------------------------

def conv_block(c_in, c_out):
    """Reference Model 6 block: Conv3x3 (He init) -> BatchNorm -> ReLU."""
    conv = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
    nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    return nn.Sequential(conv, nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class WaferCNN(nn.Module):
    """Reference Model 6 (16-32-64, MaxPool 2, GlobalMaxPool, 9-way head) + a 128 block.

    GlobalMaxPool makes the detector translation invariant, which is what Scratch / Loc need
    (their position varies from wafer to wafer, eda.md §5).
    """

    def __init__(self, widths=(16, 32, 64, 128), n_classes=1 + len(DEFECT_TYPES)):
        super().__init__()
        layers, c = [], 3
        for i, w in enumerate(widths):
            layers.append(conv_block(c, w))
            if i < len(widths) - 1:
                layers.append(nn.MaxPool2d(2))
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(c, n_classes)

    def forward(self, x):
        return self.head(self.features(x).amax((2, 3)))

    def score(self, x):
        logits = self.forward(x).float()
        return torch.logsumexp(logits[:, 1:], 1) - logits[:, 0]


class ConvAE(nn.Module):
    """Convolutional autoencoder, 64x64x3 -> latent -> 64x64x3 logits (ablation)."""

    def __init__(self, widths=(32, 64, 128), latent=64, pool=5, topk=16):
        super().__init__()
        self.pool, self.topk = pool, topk
        side = 64 // 2 ** len(widths)
        enc, c = [], 3
        for w in widths:
            enc += [conv_block(c, w), nn.MaxPool2d(2)]
            c = w
        self.encoder = nn.Sequential(*enc, nn.Flatten(), nn.Linear(c * side * side, latent))
        dec = [nn.Linear(latent, c * side * side), nn.ReLU(inplace=True), nn.Unflatten(1, (c, side, side))]
        for w in list(reversed(widths))[1:] + [widths[0]]:
            dec += [nn.ConvTranspose2d(c, w, 4, stride=2, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            c = w
        dec.append(nn.Conv2d(c, 3, 3, padding=1))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def score(self, x):
        err = F.cross_entropy(self.forward(x).float(), x.argmax(1), reduction="none")  # (B, H, W)
        err = err * (x[:, 0] == 0)  # wafer pixels only (channel 0 = background)
        err = F.avg_pool2d(err[:, None], self.pool, stride=1, padding=self.pool // 2)[:, 0]
        return err.flatten(1).topk(self.topk, 1).values.mean(1)


def build_model(args):
    if args.method == "synth":
        return WaferCNN(widths=tuple(args.widths or (16, 32, 64, 128)))
    return ConvAE(widths=tuple(args.widths or (32, 64, 128)), latent=args.latent,
                  pool=args.ae_pool, topk=args.ae_topk)


# ---- training / scoring ---------------------------------------------------------------------

def make_batch(xb, args):
    """Normal wafers -> (model input, target). Only train normals ever enter here."""
    xb = random_dihedral(xb)
    if args.method == "synth":
        n_syn = int(len(xb) * args.synth_frac)
        types = torch.zeros(len(xb), dtype=torch.long, device=xb.device)
        types[:n_syn] = torch.randint(1, len(DEFECT_TYPES) + 1, (n_syn,), device=xb.device)
        return onehot(synthesize(xb, types)), types
    return onehot(xb), xb.long()


@torch.no_grad()
def score_all(model, X, args, tta=False):
    """Anomaly scores (higher = more anomalous) for a (N, H, W) uint8 tensor on the device."""
    model.eval()
    amp = X.device.type == "cuda"
    out = []
    for i in range(0, len(X), args.eval_batch):
        xb = X[i:i + args.eval_batch]
        with torch.autocast(X.device.type, dtype=torch.float16, enabled=amp):
            views = dihedral_views(xb) if tta else [xb]
            s = torch.stack([model.score(onehot(v)) for v in views]).mean(0)
        out.append(s.float().cpu())
    return torch.cat(out).numpy()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(seed, X_tr, X_val, y_val, args):
    """Train on normals, keep the epoch with the best val PR-AUC (early stopping)."""
    set_seed(seed)
    device = X_tr.device
    amp = device.type == "cuda"
    model = build_model(args).to(device)
    steps = len(X_tr) // args.batch_size
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    history, best_ap, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(len(X_tr), device=device)
        total = torch.zeros((), device=device)
        for s in range(steps):
            inputs, target = make_batch(X_tr[perm[s * args.batch_size:(s + 1) * args.batch_size]], args)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(inputs), target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.detach()

        val_ap = average_precision_score(y_val, score_all(model, X_val, args))
        history.append({"epoch": epoch, "train_loss": (total / steps).item(), "val_pr_auc": val_ap,
                        "seconds": time.perf_counter() - t0})
        print(f"  epoch {epoch:3d}  loss {history[-1]['train_loss']:.4f}  val PR-AUC {val_ap:.4f}"
              f"  ({history[-1]['seconds']:.0f}s)")
        if val_ap > best_ap:
            best_ap, best_epoch, bad = val_ap, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop: no val improvement for {args.patience} epochs")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch, best_ap, time.perf_counter() - t_start


def save_synthetic_examples(X_normal, path, seed=0):
    """Two example wafers per class: none + the 8 synthetic defect types."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    set_seed(seed)
    rng = np.random.default_rng(seed)
    n_cls = 1 + len(DEFECT_TYPES)
    base = torch.from_numpy(X_normal[rng.choice(len(X_normal), 2 * n_cls, replace=False)])
    types = torch.arange(n_cls).repeat(2)
    x = synthesize(base, types).numpy()

    fig, axes = plt.subplots(2, n_cls, figsize=(1.5 * n_cls, 3.4))
    cmap = ListedColormap(["white", "#c8d6e5", "#c0392b"])  # background / pass / fail
    for ax, img, t in zip(axes.flat, x, types.tolist()):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title((["none"] + DEFECT_TYPES)[t], fontsize=9)
        ax.axis("off")
    fig.suptitle("Synthetic defects drawn on train-normal wafers (Tier 2 training input)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=["synth", "ae"], default="synth")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5, help="early stopping on val PR-AUC")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--widths", type=int, nargs="+", default=None,
                        help="conv widths (default synth: 16 32 64 128, ae: 32 64 128)")
    parser.add_argument("--synth-frac", type=float, default=0.5, help="share of each batch turned into synthetic defects")
    parser.add_argument("--latent", type=int, default=64, help="ae only")
    parser.add_argument("--ae-pool", type=int, default=5, help="ae only: avg-pool size of the error map")
    parser.add_argument("--ae-topk", type=int, default=16, help="ae only: mean of the k largest pooled errors")
    parser.add_argument("--tta", action="store_true", help="average the score over 8 flips/rotations")
    parser.add_argument("--eval-batch", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="DEBUG ONLY: subsample every split to N wafers")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "tier2")
    args = parser.parse_args()

    name = METHOD_NAMES[args.method]
    out = args.out / (args.method + ("_debug" if args.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{name} on {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))
    torch.backends.cudnn.benchmark = True

    X_train, y_train, _, _ = load_split("train")
    X_val, y_val, _, _ = load_split("val")
    X_test, y_test, ft_test, idx_test = load_split("test")
    X_train_normal = X_train[y_train == 0]  # Tier 2: train normals only
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

    if args.method == "synth":
        fig_path = (out if args.limit else ROOT / "figures" / "tier2") / "synthetic_examples"
        save_synthetic_examples(X_train_normal, fig_path)
        print(f"saved {fig_path}.png")

    X_tr_t = torch.from_numpy(X_train_normal).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)

    rows, recalls = [], {}
    scores_df = pd.DataFrame({"index": idx_test, "failureType": ft_test, "label": y_test})
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        model, history, best_epoch, best_val_ap, fit_seconds = train(seed, X_tr_t, X_val_t, y_val, args)
        history.to_csv(out / f"history_seed{seed}.csv", index=False, float_format="%.4f")
        torch.save(model.state_dict(), out / f"model_seed{seed}.pt")

        # tau on VAL only, then TEST scored exactly once
        tau = find_best_f1_threshold(y_val, score_all(model, X_val_t, args, tta=args.tta))
        t0 = time.perf_counter()
        scores_test = score_all(model, X_test_t, args, tta=args.tta)
        ms_per_sample = (time.perf_counter() - t0) * 1000 / len(X_test)

        m = evaluate(y_test, scores_test, tau)
        m.update({"method": f"{name}_seed{seed}", "seed": seed, "best_epoch": best_epoch,
                  "val_pr_auc": best_val_ap, "ms_per_sample": ms_per_sample, "fit_seconds": fit_seconds})
        rows.append(m)
        recalls[f"seed{seed}"] = per_type_recall(ft_test, scores_test, tau)["rate"]
        scores_df[f"score_seed{seed}"] = scores_test
        scores_df[f"pred_seed{seed}"] = (scores_test >= tau).astype(int)
        print(f"  test PR-AUC {m['pr_auc']:.4f}  ROC-AUC {m['roc_auc']:.4f}  F1 {m['f1']:.4f}"
              f"  recall {m['recall']:.4f}  FPR {m['fpr']:.4f}  (tau {tau:.4f}, best epoch {best_epoch})")

    cols = ["pr_auc", "roc_auc", "f1", "precision", "recall", "fpr", "accuracy", "tau",
            "tp", "fp", "fn", "tn", "ms_per_sample", "fit_seconds", "best_epoch", "val_pr_auc"]
    metrics = pd.DataFrame(rows).set_index("method")[cols]
    seed_rows = metrics.copy()
    metrics.loc[f"{name}_mean"] = seed_rows.mean()
    metrics.loc[f"{name}_std"] = seed_rows.std(ddof=0) if len(seed_rows) > 1 else 0.0
    metrics.to_csv(out / "metrics.csv", float_format="%.4f")

    per_type = pd.DataFrame(recalls)
    per_type["mean"] = per_type.mean(axis=1)
    per_type["std"] = per_type[list(recalls)].std(axis=1, ddof=0)
    per_type = per_type.join(pd.Series(ft_test).value_counts().rename("n_test"))
    order = ["none"] + per_type.drop(index="none").sort_values("mean").index.tolist()
    per_type.loc[order].to_csv(out / "per_type_recall.csv", float_format="%.4f")
    scores_df.to_csv(out / "scores_test.csv", index=False, float_format="%.6f")

    pd.set_option("display.width", 200)
    print("\nTest metrics")
    print(metrics[["pr_auc", "roc_auc", "f1", "recall", "fpr", "ms_per_sample"]].round(4).to_string(), "\n")
    print("Per-type detection rate on test ('none' row = false alarm rate)")
    print(per_type.loc[order].round(3).to_string(), "\n")
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()

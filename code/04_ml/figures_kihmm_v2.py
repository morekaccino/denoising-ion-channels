"""Performance figures for KI-HMM v2 on the frozen synth_v2 test set.

Saves:
  results/figures/kihmm_v2_predictions.png   traces + open-count predictions
  results/figures/kihmm_v2_state_shares.png  per-state counts + N confusion
  results/figures/kihmm_v2_rate_scatter.png  effective rate recovery + per-rate R2

Usage:
  python code/04_ml/figures_kihmm_v2.py --model code/04_ml/models/kihmm_v2_v2a.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code" / "04_ml"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import benchmark_v2 as B2  # noqa: E402
import torch_models_v2 as T  # noqa: E402
from eval_kihmm_v2 import effective_params, predict_rates  # noqa: E402
from train_kihmm_v2 import batch_tensors, group_batches  # noqa: E402

RESULTS = ROOT / "code" / "04_ml" / "results"
FIGURES = RESULTS / "figures"


def collect(model, data, per_batch: int = 16):
    X, y, r, N, group, R, K = data
    counts, n_hat, ys, rs = [], [], [], []
    model.eval()
    with torch.no_grad():
        for gids in group_batches(len(R), per_batch, np.random.default_rng(0), shuffle=False):
            idx = (np.asarray(gids)[:, None] * K + np.arange(K)[None, :]).reshape(-1)
            xb, rb, Nb, Rb, gb = batch_tensors(X, r, N, R, K, gids)
            out = T.predict(model, xb, gb)
            counts.append(out["counts"].cpu().numpy())
            n_hat.append(out["N_hat"].cpu().numpy())
            ys.append(y[idx])
            rs.append(r[idx])
    return (np.concatenate(counts), np.concatenate(n_hat).astype(int),
            np.concatenate(ys), np.concatenate(rs))


def r2(pred, truth):
    return float(1 - ((pred - truth) ** 2).mean() / max(truth.var(), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "code" / "04_ml" / "models" / "kihmm_v2_v2a.pt"))
    parser.add_argument("--scale", default="1")
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location=T.DEVICE)
    model = T.KIHMMv2(hidden=ckpt["args"].get("hidden", 64)).to(T.DEVICE)
    model.load_state_dict(ckpt["model"])

    d = B2.load_split("test")
    X = d[f"X_s{args.scale}"]
    data = (X, d["y"], d["r"], d["N"], d["group"], d["R"], int((d["group"] == 0).sum()))
    counts, n_hat, y, r = collect(model, data)
    counts_true = np.stack([(r == s).sum(axis=1) for s in range(7)], axis=1).astype(float)
    N_true = d["N"]
    FIGURES.mkdir(parents=True, exist_ok=True)

    open_idx = np.flatnonzero(B2.B.STATEMAP == 1)
    open_pred = counts[:, open_idx].sum(axis=1)
    open_acc = (np.round(open_pred) == y).mean()
    print(f"traces={len(X)}  N_acc={(n_hat == N_true).mean():.3f}  "
          f"state_mae={np.abs(counts - counts_true).mean():.3f}  open_acc={open_acc:.3f}")

    # ---- figure 1: traces + open-count prediction ----
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for ax, n_val in zip(axes, (1, 2, 3)):
        acc = (np.round(open_pred) == y).mean(axis=1)
        cand = np.flatnonzero(N_true == n_val)
        i = cand[np.argsort(acc[cand])[len(cand) // 2]]
        ax.plot(X[i], color="lightsteelblue", lw=0.8)
        ax.set_ylabel("current", color="steelblue")
        ax2 = ax.twinx()
        ax2.step(np.arange(len(y[i])), y[i], color="green", ls="--", lw=2.0, where="post",
                 label="true open count")
        ax2.step(np.arange(len(y[i])), open_pred[i], color="red", lw=1.6, where="post",
                 label="predicted (expected)")
        ax2.set_ylim(-0.3, max(4, n_val + 1.5))
        ax2.set_yticks(range(0, n_val + 2))
        ax2.set_ylabel("open channels", color="darkred")
        ax.set_title(
            f"true N={N_true[i]}, predicted N={n_hat[i]}, "
            f"open-count accuracy={acc[i]:.3f}, "
            f"state-count MAE={np.abs(counts[i] - counts_true[i]).mean():.3f}",
            fontsize=10,
        )
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("time (samples, 100 Hz)")
    fig.suptitle(f"KI-HMM v2 predictions vs truth (frozen test, noise x{args.scale})", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "kihmm_v2_predictions.png", dpi=130)
    print("saved", FIGURES / "kihmm_v2_predictions.png")

    # ---- figure 2: state shares + per-state MAE + N confusion ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    names = ["C1a", "C1b", "C2", "O1", "O2", "C3", "C4"]
    share_true = counts_true.mean(axis=(0, 2))
    share_pred = counts.mean(axis=(0, 2))
    w = 0.38
    axes[0].bar(np.arange(7) - w / 2, share_true, w, label="true")
    axes[0].bar(np.arange(7) + w / 2, share_pred, w, label="predicted")
    axes[0].set_xticks(range(7), names)
    axes[0].set_ylabel("mean channels per state")
    axes[0].set_title("average per-state counts")
    axes[0].legend()

    mae = np.abs(counts - counts_true).mean(axis=(0, 2))
    axes[1].bar(range(7), mae)
    axes[1].set_xticks(range(7), names)
    axes[1].set_ylabel("MAE (channels)")
    axes[1].set_title("per-state count error")

    cm = np.zeros((5, 5))
    for t, p in zip(N_true, n_hat):
        cm[int(t) - 1, int(p) - 1] += 1
    im = axes[2].imshow(cm, cmap="Blues")
    for (a, b), v in np.ndenumerate(cm):
        axes[2].text(b, a, int(v), ha="center", va="center", fontsize=8,
                     color="white" if v > cm.max() / 2 else "black")
    axes[2].set_xticks(range(5), range(1, 6))
    axes[2].set_yticks(range(5), range(1, 6))
    axes[2].set_xlabel("predicted N")
    axes[2].set_ylabel("true N")
    axes[2].set_title(f"N confusion (acc={np.mean(N_true == n_hat):.3f})")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGURES / "kihmm_v2_state_shares.png", dpi=130)
    print("saved", FIGURES / "kihmm_v2_state_shares.png")

    # ---- figure 3: effective rate recovery + per-rate R2 ----
    pred_log, true_log = predict_rates(model, data)
    pred_r, true_r = np.exp(pred_log), np.exp(true_log)
    eff_p, eff_t = effective_params(pred_r), effective_params(true_r)
    F = np.asarray(json.loads((RESULTS / "fisher_synth_v2.json").read_text())["fisher"])
    evals, evecs = np.linalg.eigh(F)
    order = np.argsort(evals)[::-1]
    v1 = evecs[:, order[0]]
    if np.corrcoef(true_log @ v1, pred_log @ v1)[0, 1] < 0:
        v1 = -v1

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    panels = [
        (eff_t[:, 0], eff_p[:, 0], "log opening rate"),
        (eff_t[:, 1], eff_p[:, 1], "log closing rate"),
        (eff_t[:, 2], eff_p[:, 2], "p_open"),
        (true_log @ v1, pred_log @ v1, "top identifiable direction"),
    ]
    for ax, (tx, py, title) in zip(axes.ravel(), panels):
        ax.scatter(tx, py, s=18)
        lo, hi = min(tx.min(), py.min()), max(tx.max(), py.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
        ax.set_title(f"{title}: R2={r2(py, tx):.2f}")
    fig.suptitle(f"Markov-parameter recovery per group (test x{args.scale}, 64 groups)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "kihmm_v2_rate_scatter.png", dpi=130)
    print("saved", FIGURES / "kihmm_v2_rate_scatter.png")

    per_rate = [r2(pred_log[:, j], true_log[:, j]) for j in range(12)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    axes[0].bar(range(12), per_rate)
    axes[0].set_xticks(range(12), B2.RATE_NAMES, rotation=60, fontsize=8)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("R2")
    axes[0].set_title("per-rate R2 (most rates are not identifiable)")
    axes[1].semilogy(np.sort(evals)[::-1], "o-")
    axes[1].axhline(1 / 0.347**2, color="r", ls="--",
                    label="prior-only level (0.347)")
    axes[1].set_xlabel("direction")
    axes[1].set_ylabel("Fisher eigenvalue (log)")
    axes[1].set_title("how much rate info the signal carries")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "kihmm_v2_rate_recovery.png", dpi=130)
    print("saved", FIGURES / "kihmm_v2_rate_recovery.png")

    print(f"effective R2: opening={r2(eff_p[:,0], eff_t[:,0]):.3f} "
          f"closing={r2(eff_p[:,1], eff_t[:,1]):.3f} "
          f"p_open={r2(eff_p[:,2], eff_t[:,2]):.3f}")


if __name__ == "__main__":
    main()

"""Performance figures for KI-HMM v4 on the frozen synth_v2 test set.

Saves:
  results/figures/kihmm_v4_predictions.png    traces + open-count posterior
  results/figures/kihmm_v4_state_shares.png   per-state counts + N confusion
  results/figures/kihmm_v4_rate_scatter.png   effective rate recovery
  results/figures/kihmm_v4_rate_recovery.png  per-rate R2 + Fisher spectrum

Usage:
  python code/04_ml/figures_kihmm_v4.py --model code/04_ml/models/kihmm_v4_v4a.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "04_ml"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import benchmark_v2 as B2  # noqa: E402
import torch_models_v4 as V4  # noqa: E402
from eval_kihmm_v2 import effective_params, r2  # noqa: E402
from eval_kihmm_v4 import load_model, predict_rates  # noqa: E402
from train_kihmm_v4 import batch, load  # noqa: E402
from train_kihmm_v2 import group_batches  # noqa: E402

RESULTS = ROOT / "code" / "04_ml" / "results"
FIGURES = RESULTS / "figures"
STATE_NAMES = ["C1a", "C1b", "C2", "O1", "O2", "C3", "C4"]


@torch.no_grad()
def collect(model, data: dict, device: str, per_batch: int = 8):
    counts, e_open, n_hat = [], [], []
    for gids in group_batches(len(data["R"]), per_batch, np.random.default_rng(0), shuffle=False):
        b = batch(data, gids, device)
        p = V4.predict(model, b["x"], b["group"], emission_chunk=250)
        counts.append(p["counts"].cpu().numpy())
        e_open.append(p["e_open"].cpu().numpy())
        n_hat.append(p["N_hat"].cpu().numpy())
    return np.concatenate(counts), np.concatenate(e_open), np.concatenate(n_hat).astype(int)


def figure_predictions(X, y, N_true, n_hat, e_open, counts, counts_true, scale: str, prefix: str) -> None:
    per_trace = (np.round(e_open) == y).mean(axis=1)
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for ax, n_val in zip(axes, (1, 2, 3)):
        cand = np.flatnonzero(N_true == n_val)
        i = cand[np.argsort(per_trace[cand])[len(cand) // 2]]
        ax.plot(X[i], color="lightsteelblue", lw=0.8, label="signal")
        ax.set_ylabel("current", color="steelblue")
        ax2 = ax.twinx()
        ax2.step(np.arange(len(y[i])), y[i], color="green", ls="--", lw=2.0, where="post",
                 label="true open count")
        ax2.step(np.arange(len(y[i])), e_open[i], color="red", lw=1.4, where="post",
                 label="posterior expected open count")
        ax2.set_ylim(-0.3, max(4, n_val + 1.5))
        ax2.set_yticks(range(0, n_val + 2))
        ax2.set_ylabel("open channels", color="darkred")
        ax.set_title(f"true N={N_true[i]}, predicted N={n_hat[i]}, "
                     f"open-count accuracy={per_trace[i]:.3f}, "
                     f"state-count MAE={np.abs(counts[i] - counts_true[i]).mean():.3f}", fontsize=10)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("time (samples, 100 Hz)")
    fig.suptitle(f"{prefix} predictions vs truth (frozen test, noise x{scale})", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{prefix}_predictions.png", dpi=130)
    print("saved", FIGURES / f"{prefix}_predictions.png")


def figure_state_shares(counts, counts_true, N_true, n_hat, prefix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    w = 0.38
    axes[0].bar(np.arange(7) - w / 2, counts_true.mean(axis=(0, 2)), w, label="true")
    axes[0].bar(np.arange(7) + w / 2, counts.mean(axis=(0, 2)), w, label="predicted")
    axes[0].set_xticks(range(7), STATE_NAMES)
    axes[0].set_ylabel("mean channels per state")
    axes[0].set_title("average per-state counts")
    axes[0].legend()

    axes[1].bar(range(7), np.abs(counts - counts_true).mean(axis=(0, 2)))
    axes[1].set_xticks(range(7), STATE_NAMES)
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
    fig.savefig(FIGURES / f"{prefix}_state_shares.png", dpi=130)
    print("saved", FIGURES / f"{prefix}_state_shares.png")


def figure_rates(pred_log, true_log, scale: str, prefix: str) -> None:
    eff_p, eff_t = effective_params(np.exp(pred_log)), effective_params(np.exp(true_log))
    evals, evecs = np.linalg.eigh(
        np.asarray(json.loads((RESULTS / "fisher_synth_v2.json").read_text())["fisher"]))
    order = np.argsort(evals)[::-1]
    v1 = evecs[:, order[0]]
    if np.corrcoef(true_log @ v1, pred_log @ v1)[0, 1] < 0:
        v1 = -v1

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    panels = [(eff_t[:, 0], eff_p[:, 0], "log opening rate"),
              (eff_t[:, 1], eff_p[:, 1], "log closing rate"),
              (eff_t[:, 2], eff_p[:, 2], "p_open"),
              (true_log @ v1, pred_log @ v1, "top identifiable direction")]
    for ax, (tx, py, title) in zip(axes.ravel(), panels):
        ax.scatter(tx, py, s=18)
        lo, hi = min(tx.min(), py.min()), max(tx.max(), py.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
        ax.set_title(f"{title}: R2={r2(py, tx):.2f}")
    fig.suptitle(f"{prefix} Markov-parameter recovery per group (test x{scale})", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{prefix}_rate_scatter.png", dpi=130)
    print("saved", FIGURES / f"{prefix}_rate_scatter.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    axes[0].bar(range(12), [r2(pred_log[:, j], true_log[:, j]) for j in range(12)])
    axes[0].set_xticks(range(12), B2.RATE_NAMES, rotation=60, fontsize=8)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("R2")
    axes[0].set_title("per-rate R2 (most rates are not identifiable)")
    axes[1].semilogy(np.sort(evals)[::-1], "o-")
    axes[1].axhline(1 / 0.347**2, color="r", ls="--", label="prior-only level (0.347)")
    axes[1].set_xlabel("direction")
    axes[1].set_ylabel("Fisher eigenvalue (log)")
    axes[1].set_title("how much rate info the signal carries")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"{prefix}_rate_recovery.png", dpi=130)
    print("saved", FIGURES / f"{prefix}_rate_recovery.png")
    print(f"effective R2: opening={r2(eff_p[:,0], eff_t[:,0]):.3f} "
          f"closing={r2(eff_p[:,1], eff_t[:,1]):.3f} p_open={r2(eff_p[:,2], eff_t[:,2]):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "code" / "04_ml" / "models" / "kihmm_v4_v4a.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scale", default="1")
    parser.add_argument("--prefix", default="kihmm_v4")
    args = parser.parse_args()

    model = load_model(args.model, args.device)
    data = load("test", args.scale)
    counts, e_open, n_hat = collect(model, data, args.device)
    counts_true = np.stack([(data["r"] == s).sum(axis=1) for s in range(7)], axis=1).astype(float)
    FIGURES.mkdir(parents=True, exist_ok=True)
    print(f"traces={len(data['X'])} N_acc={(n_hat == data['N']).mean():.3f} "
          f"state_mae={np.abs(counts - counts_true).mean():.3f} "
          f"open_acc={(np.round(e_open) == data['y']).mean():.3f}")

    figure_predictions(data["X"], data["y"], data["N"], n_hat, e_open, counts, counts_true,
                       args.scale, args.prefix)
    figure_state_shares(counts, counts_true, data["N"], n_hat, args.prefix)
    pred_log, true_log = predict_rates(model, data, args.device)
    figure_rates(pred_log, true_log, args.scale, args.prefix)


if __name__ == "__main__":
    main()

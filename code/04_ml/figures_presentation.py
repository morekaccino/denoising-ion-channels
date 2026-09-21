"""Presentation figures for the KI-HMM v5 writeup.

Everything the Notion page needs that the repo did not already have, in one
place: a five-panel walk from one channel to five, the architecture, the
problem and the generative model, the model ladder, and the bake-off
leaderboards. Existing result figures are reused rather than regenerated.

Saves to ``code/04_ml/results/figures/``.

Usage:
  python code/04_ml/figures_presentation.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "04_ml"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import torch  # noqa: E402
import benchmark as B  # noqa: E402
import torch_models_v4 as V4  # noqa: E402
from eval_kihmm_v4 import load_model  # noqa: E402
from train_kihmm_v4 import batch, load  # noqa: E402
from train_kihmm_v2 import group_batches  # noqa: E402

RESULTS = ROOT / "code" / "04_ml" / "results"
FIGURES = RESULTS / "figures"
MODEL = ROOT / "code" / "04_ml" / "models" / "kihmm_v4_v5a.pt"
STATE_NAMES = ["C1a", "C1b", "C2", "O1", "O2", "C3", "C4"]


def figure_n_ladder(model, data: dict) -> None:
    """One row per channel count: raw trace, true vs predicted open count, per-state counts."""
    X, y, r, N = data["X"], data["y"], data["r"], data["N"]
    counts_all, e_open, n_hat = [], [], []
    for gids in group_batches(len(data["R"]), 8, np.random.default_rng(0), shuffle=False):
        b = batch(data, gids, "cpu")
        p = V4.predict(model, b["x"], b["group"], emission_chunk=250)
        counts_all.append(p["counts"].cpu().numpy())
        e_open.append(p["e_open"].cpu().numpy())
        n_hat.append(p["N_hat"].cpu().numpy())
    counts, e_open, n_hat = np.concatenate(counts_all), np.concatenate(e_open), np.concatenate(n_hat)

    fig, axes = plt.subplots(5, 3, figsize=(19, 15), gridspec_kw={"width_ratios": [3, 3, 2.2]})
    for row, n_val in enumerate(range(1, 6)):
        cand = np.flatnonzero(N == n_val)
        acc = (np.round(e_open[cand]) == y[cand]).mean(axis=1)
        i = cand[np.argsort(acc)[len(cand) // 2]]
        t = np.arange(X.shape[1])

        axes[row, 0].plot(t, X[i], color="steelblue", lw=0.7)
        axes[row, 0].set_ylabel(f"N = {n_val}", fontsize=11, fontweight="bold")
        axes[row, 0].set_yticks([])
        if row == 0:
            axes[row, 0].set_title("raw recording (summed current)", fontsize=11)

        axes[row, 1].plot(t, X[i], color="lightsteelblue", lw=0.6, alpha=0.6)
        ax2 = axes[row, 1].twinx()
        ax2.step(t, y[i], color="green", ls="--", lw=2.0, where="post", label="true")
        ax2.step(t, e_open[i], color="red", lw=1.5, where="post", label="predicted")
        ax2.set_ylim(-0.3, n_val + 1.2)
        ax2.set_yticks(range(n_val + 1))
        axes[row, 1].set_yticks([])
        if row == 0:
            axes[row, 1].set_title("open channels: truth vs model", fontsize=11)
            ax2.legend(loc="upper right", fontsize=8)
        axes[row, 1].text(0.01, 0.93, f"accuracy {acc[np.flatnonzero(cand == i)[0]]:.3f}  (N predicted {n_hat[i]})",
                          transform=axes[row, 1].transAxes, fontsize=8, va="top")

        bottom = np.zeros(X.shape[1])
        for s in range(7):
            axes[row, 2].fill_between(t, bottom, bottom + counts[i, s], step="post",
                                      alpha=0.85, label=STATE_NAMES[s] if row == 0 else None)
            bottom += counts[i, s]
        axes[row, 2].set_ylim(0, n_val + 0.3)
        axes[row, 2].set_yticks(range(n_val + 1))
        if row == 0:
            axes[row, 2].set_title("channels per kinetic state", fontsize=11)
            axes[row, 2].legend(loc="upper right", fontsize=7, ncol=4)
    for ax in axes[-1]:
        ax.set_xlabel("time (samples, 100 Hz)")
    fig.suptitle("KI-HMM v5 on the frozen test set, one to five channels", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(FIGURES / "present_n_ladder.png", dpi=120)
    plt.close(fig)
    print("saved", FIGURES / "present_n_ladder.png")


def figure_architecture() -> None:
    """The network in plain blocks, with the inference loop drawn in."""
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)

    def box(x, y, w, h, text, fc, fs=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.2",
                                    fc=fc, ec="#333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, dashed=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=16, lw=1.4, color="#333",
                                     linestyle="--" if dashed else "-"))

    box(2, 26, 12, 8, "summed\ntrace", "#dbeafe")
    box(20, 26, 16, 8, "TCN encoder\n(sees the shape)", "#e0e7ff")
    box(42, 40, 18, 7, "N head\n+ amplitude histogram", "#fef3c7", 9)
    box(42, 26, 18, 7, "learned noise density\n(pointwise emission)", "#dcfce7", 9)
    box(42, 12, 18, 7, "rate head\n(group of traces)", "#fee2e2", 9)
    box(66, 26, 15, 8, "exact chain\nmatrix_exp(dt*R)", "#fae8ff", 9)
    box(86, 26, 12, 8, "forward\nbackward", "#fce7f3", 9)

    arrow(14, 30, 20, 30)
    arrow(36, 30, 42, 30)
    arrow(36, 30, 42, 43.5)
    arrow(36, 30, 42, 15.5)
    arrow(60, 15.5, 66, 28)
    arrow(60, 43.5, 66, 31)
    arrow(60, 30, 66, 30)
    arrow(81, 30, 86, 30)

    box(70, 44, 26, 5, "N by model evidence\n(score each candidate, take the best)", "#fef9c3", 9)
    box(70, 6, 26, 5, "rates by likelihood\n(Adam on log evidence)", "#fef9c3", 9)
    arrow(86, 34, 78, 44, dashed=True)
    arrow(78, 11, 66, 24, dashed=True)
    ax.text(50, 57, "KI-HMM v5: one network, exact inference", ha="center", fontsize=14,
            fontweight="bold")
    ax.text(50, 2, "dashed: the inference loop that replaces the heads at test time",
            ha="center", fontsize=9, style="italic", color="#555")
    fig.tight_layout()
    fig.savefig(FIGURES / "present_architecture.png", dpi=130)
    plt.close(fig)
    print("saved", FIGURES / "present_architecture.png")


def figure_problem(data: dict) -> None:
    """The measurement problem: one trace, many hidden channels."""
    i = int(np.flatnonzero(data["N"] == 3)[0])
    X, r = data["X"][i], data["r"][i]
    t = np.arange(200)
    fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.6]})
    for ch in range(3):
        single = B.STATEMAP[r[ch][:200]].astype(float)
        axes[ch].step(t, single, color="#0e7490", lw=1.6, where="post")
        axes[ch].set_ylabel(f"ch {ch + 1}", rotation=0, ha="right", va="center")
        axes[ch].set_yticks([0, 1])
        axes[ch].set_yticklabels(["closed", "open"], fontsize=8)
    axes[3].plot(t, X[:200], color="#1e293b", lw=0.9)
    axes[3].set_ylabel("what we record", rotation=0, ha="right", va="center")
    axes[3].set_xlabel("time (samples)")
    fig.suptitle("Three channels open and close on their own. We only see the sum, plus noise.",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "present_problem.png", dpi=130)
    plt.close(fig)
    print("saved", FIGURES / "present_problem.png")


def figure_ladder() -> None:
    """The model ladder on the frozen test set."""
    rows = [("v2a", 0.605, 0.537, 0.393), ("v3a", 0.657, 0.592, 0.472),
            ("v4b", 0.815, 0.718, 0.469), ("v5a feedforward", 0.827, 0.729, 0.508),
            ("v5a + exact inference", 0.853, 0.755, 0.508),
            ("true N + true rates", 0.928, 0.695, 0.451)]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(rows))
    w = 0.26
    for j, (lab, col) in enumerate(zip(("noise x1", "noise x2", "noise x4"),
                                       ("#1d4ed8", "#7c3aed", "#be185d"))):
        ax.bar(x + (j - 1) * w, [r[1 + j] for r in rows], w, label=lab, color=col)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=9)
    ax.set_ylabel("open-count accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Each generation of the method, on the same frozen test set", fontsize=12)
    ax.legend()
    for j in range(3):
        for i, r in enumerate(rows):
            ax.text(i + (j - 1) * w, r[1 + j] + 0.012, f"{r[1 + j]:.2f}",
                    ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "present_ladder.png", dpi=130)
    plt.close(fig)
    print("saved", FIGURES / "present_ladder.png")


def figure_bakeoff() -> None:
    """The bake-off leaderboards."""
    count = json.loads((RESULTS / "bakeoff_count.json").read_text())
    rates = json.loads((RESULTS / "bakeoff_rates.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    items = [(k, v["test"]) for k, v in count.items() if isinstance(v.get("test"), float)]
    items.sort(key=lambda kv: kv[1])
    names = [k.replace("MLP on ", "").replace("reference: ", "") for k, _ in items]
    colors = ["#94a3b8" if "reference" in k or "v4b" in k else "#1d4ed8" for k, _ in items]
    axes[0].barh(range(len(items)), [v for _, v in items], color=colors)
    axes[0].set_yticks(range(len(items)), names, fontsize=8)
    axes[0].set_xlabel("test accuracy")
    axes[0].set_xlim(0.8, 1.0)
    axes[0].set_title("Count head bake-off (3x training data)", fontsize=11)
    for i, (_, v) in enumerate(items):
        axes[0].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7)

    items2 = [(k, v["test"][0]) for k, v in rates.items() if isinstance(v.get("test"), list)]
    items2.sort(key=lambda kv: kv[1])
    names2 = [k.replace("reference: ", "") for k, _ in items2]
    colors2 = ["#94a3b8" if "v4b" in k else "#7c3aed" for k, _ in items2]
    axes[1].barh(range(len(items2)), [v for _, v in items2], color=colors2)
    axes[1].set_yticks(range(len(items2)), names2, fontsize=8)
    axes[1].set_xlabel("top identifiable direction R2")
    axes[1].axvline(0, color="k", lw=0.6)
    axes[1].set_title("Rate head bake-off (3x training data)", fontsize=11)
    for i, (_, v) in enumerate(items2):
        axes[1].text(v + (0.02 if v >= 0 else -0.02), i, f"{v:.2f}",
                     va="center", ha="left" if v >= 0 else "right", fontsize=7)
    fig.suptitle("Architecture barely matters; data and the optimiser do", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES / "present_bakeoff.png", dpi=120)
    plt.close(fig)
    print("saved", FIGURES / "present_bakeoff.png")


def figure_generative(data: dict) -> None:
    """The generative model: states, counts, levels, noise."""
    i = int(np.flatnonzero(data["N"] == 2)[0])
    r, X, y = data["r"][i], data["X"][i], data["y"][i]
    t = np.arange(300)
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1, 1, 1.4]})
    axes[0].imshow(r[:, :300], aspect="auto", cmap="tab10", interpolation="nearest")
    axes[0].set_ylabel("hidden state\nper channel", rotation=0, ha="right", va="center", fontsize=9)
    axes[0].set_yticks([])
    axes[0].set_title("each channel walks the 7-state CFTR graph", fontsize=10)

    axes[1].step(t, y[:300], color="green", lw=1.8, where="post")
    axes[1].set_ylabel("open count", rotation=0, ha="right", va="center", fontsize=9)
    axes[1].set_yticks(range(3))
    axes[1].set_title("only the count of open channels shapes the signal", fontsize=10)

    level = y[:300] * 1.4 + (2 - y[:300]) * 0.58
    axes[2].step(t, level, color="#0e7490", lw=1.8, where="post")
    axes[2].set_ylabel("clean level", rotation=0, ha="right", va="center", fontsize=9)
    axes[2].set_title("k open channels sit at one level", fontsize=10)

    axes[3].plot(t, X[:300], color="#1e293b", lw=0.8)
    axes[3].set_ylabel("recorded", rotation=0, ha="right", va="center", fontsize=9)
    axes[3].set_title("plus heavy-tailed noise", fontsize=10)
    axes[3].set_xlabel("time (samples, 100 Hz)")
    fig.suptitle("The generative model the network learns to invert", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "present_generative.png", dpi=130)
    plt.close(fig)
    print("saved", FIGURES / "present_generative.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    model = load_model(str(MODEL), "cpu")
    data = load("test", "1")
    figure_n_ladder(model, data)
    figure_architecture()
    figure_problem(data)
    figure_generative(data)
    figure_ladder()
    figure_bakeoff()


if __name__ == "__main__":
    main()

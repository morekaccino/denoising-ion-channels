"""Direct predicted-versus-true figures for the KI-HMM v5 presentation.

The older presentation figures mixed model history, broad summaries and plots
that did not put predictions beside known values.  This script uses one result
set only: KI-HMM v5a with exact inference on the frozen synthetic test set.

Run ``presentation_results.py`` first.

Usage:
  python code/04_ml/figures_presentation_v5.py
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
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from eval_kihmm_v2 import effective_params  # noqa: E402

RESULTS = ROOT / "code" / "04_ml" / "results"
FIGURES = RESULTS / "figures"
CACHE = ROOT / "data" / "derived" / "presentation_v5.npz"
METRICS = RESULTS / "presentation_v5_metrics.json"
STATE_NAMES = ["C1a", "C1b", "C2", "O1", "O2", "C3", "C4"]
STATE_LETTERS = list("abcdefg")
BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#15803d"
PURPLE = "#7c3aed"


def load_results() -> tuple[dict[str, np.ndarray], dict]:
    with np.load(CACHE) as src:
        data = {key: src[key] for key in src.files}
    return data, json.loads(METRICS.read_text())


def true_counts(r: np.ndarray) -> np.ndarray:
    return np.stack([(r == s).sum(axis=1) for s in range(7)], axis=1).astype(float)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIGURES / name)


def summary_figure(metrics: dict) -> None:
    total = metrics["total"]
    rates = metrics["rates"]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis("off")
    cards = [
        ("N accuracy", total["n_accuracy"], "fraction of traces with the right channel count"),
        ("Open count accuracy", total["open_accuracy"], "fraction of samples with the right open count"),
        ("State count accuracy", total["state_rounded_accuracy"], "rounded a through g counts"),
        ("Top rate R²", rates["direction_r2"][0], "best identifiable rate direction"),
    ]
    for i, (title, value, note) in enumerate(cards):
        x = i * 0.245 + 0.02
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.18),
                0.21,
                0.64,
                boxstyle="round,pad=0.015",
                facecolor=["#dbeafe", "#dcfce7", "#f3e8ff", "#fee2e2"][i],
                edgecolor="#334155",
                transform=ax.transAxes,
            )
        )
        ax.text(x + 0.105, 0.68, title, ha="center", transform=ax.transAxes, fontsize=11)
        ax.text(
            x + 0.105,
            0.44,
            f"{100 * value:.1f}%" if "R²" not in title else f"{value:.3f}",
            ha="center",
            transform=ax.transAxes,
            fontsize=27,
            fontweight="bold",
        )
        ax.text(x + 0.105, 0.27, note, ha="center", transform=ax.transAxes, fontsize=7.5)
    ax.set_title(
        "KI-HMM v5a with exact inference, 384,000 synthetic test samples",
        fontsize=14,
        pad=12,
    )
    save(fig, "v5_summary.png")


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)

    def box(x, y, w, h, text, color, size=9):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=1",
                facecolor=color,
                edgecolor="#334155",
                linewidth=1.2,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, color="#334155"
            )
        )

    box(2, 25, 12, 9, "summed\ncurrent", "#dbeafe")
    box(20, 25, 15, 9, "temporal\nencoder", "#e0e7ff")
    box(41, 43, 17, 8, "channel count\nhead", "#fef3c7")
    box(41, 26, 17, 8, "learned signal\nand noise model", "#dcfce7")
    box(41, 9, 17, 8, "rate head\n(group of traces)", "#fee2e2")
    box(64, 25, 15, 9, "exact CFTR\nkinetic chain", "#fae8ff")
    box(85, 25, 13, 9, "forward and\nbackward pass", "#fce7f3")
    box(67, 44, 28, 7, "choose N by model evidence", "#fef9c3")
    box(67, 8, 28, 7, "improve rates by model likelihood", "#fef9c3")

    arrow(14, 29.5, 20, 29.5)
    arrow(35, 29.5, 41, 47)
    arrow(35, 29.5, 41, 30)
    arrow(35, 29.5, 41, 13)
    arrow(58, 47, 64, 32)
    arrow(58, 30, 64, 29.5)
    arrow(58, 13, 64, 27)
    arrow(79, 29.5, 85, 29.5)
    arrow(91, 34, 82, 44)
    arrow(82, 15, 70, 25)

    ax.text(50, 57, "What the model does", ha="center", fontsize=15, fontweight="bold")
    ax.text(
        50,
        2.5,
        "Outputs: N, open count, seven state counts a through g, and identifiable rates",
        ha="center",
        fontsize=10,
    )
    save(fig, "v5_architecture.png")


def representative_index(data: dict[str, np.ndarray], n: int) -> int:
    mask = (data["N_true"] == n) & (data["N_pred"] == n)
    idx = np.flatnonzero(mask)
    acc = (
        np.round(data["open_expected"][idx]).astype(int) == data["y"][idx]
    ).mean(axis=1)
    return int(idx[np.argsort(acc)[len(idx) // 2]])


def example_figure(data: dict[str, np.ndarray], n: int) -> None:
    i = representative_index(data, n)
    truth = true_counts(data["r"])[i]
    predicted = data["counts_pred"][i]
    open_round = np.round(data["open_expected"][i]).astype(int)
    correct = open_round == data["y"][i]
    open_acc = correct.mean()
    state_mae = np.abs(predicted - truth).mean()
    t = np.arange(data["X"].shape[1])

    fig = plt.figure(figsize=(15, 11))
    grid = fig.add_gridspec(5, 1, height_ratios=[1.2, 1.4, 0.28, 1, 1], hspace=0.32)
    ax0 = fig.add_subplot(grid[0])
    ax0.plot(t, data["X"][i], color="#475569", linewidth=0.7)
    ax0.set_ylabel("current")
    ax0.set_title(f"Raw synthetic recording, true N = {n}, predicted N = {data['N_pred'][i]}")

    ax1 = fig.add_subplot(grid[1], sharex=ax0)
    ax1.step(t, data["y"][i], where="post", color=GREEN, linewidth=1.8, label="true")
    ax1.step(
        t,
        data["open_expected"][i],
        where="post",
        color=RED,
        linewidth=1.2,
        label="predicted",
    )
    ax1.set_ylabel("open channels")
    ax1.set_yticks(range(n + 1))
    ax1.legend(loc="upper right")
    ax1.set_title(f"Open count: {100 * open_acc:.1f}% exact accuracy")

    ax2 = fig.add_subplot(grid[2], sharex=ax0)
    ax2.imshow(
        correct[None],
        aspect="auto",
        cmap=ListedColormap(["#ef4444", "#22c55e"]),
        vmin=0,
        vmax=1,
    )
    ax2.set_yticks([0], ["wrong / right"])
    ax2.set_xticks([])

    vmax = max(n, 1)
    ax3 = fig.add_subplot(grid[3], sharex=ax0)
    im = ax3.imshow(truth, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
    ax3.set_yticks(range(7), [f"{letter}: {name}" for letter, name in zip(STATE_LETTERS, STATE_NAMES)])
    ax3.set_title("True channel count in each state")
    fig.colorbar(im, ax=ax3, fraction=0.012, pad=0.01)

    ax4 = fig.add_subplot(grid[4], sharex=ax0)
    im = ax4.imshow(predicted, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
    ax4.set_yticks(range(7), [f"{letter}: {name}" for letter, name in zip(STATE_LETTERS, STATE_NAMES)])
    ax4.set_title(f"Predicted state counts: MAE {state_mae:.3f} channels")
    ax4.set_xlabel("time (samples at 100 Hz)")
    fig.colorbar(im, ax=ax4, fraction=0.012, pad=0.01)
    fig.suptitle(f"Predicted values beside true values, N = {n}", fontsize=15, y=0.995)
    save(fig, f"v5_example_n{n}.png")


def open_confusion_figure(data: dict[str, np.ndarray]) -> None:
    predicted = np.round(data["open_expected"]).astype(int)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for n, ax in enumerate(axes, start=1):
        mask = data["N_true"] == n
        cm = np.zeros((n + 1, n + 1), dtype=int)
        for truth, pred in zip(data["y"][mask].reshape(-1), predicted[mask].reshape(-1)):
            cm[int(truth), int(np.clip(pred, 0, n))] += 1
        pct = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(pct, vmin=0, vmax=1, cmap="Blues")
        for row in range(n + 1):
            for col in range(n + 1):
                if cm[row, col]:
                    ax.text(
                        col,
                        row,
                        f"{100 * pct[row, col]:.0f}%\n{cm[row, col]:,}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if pct[row, col] > 0.55 else "black",
                    )
        ax.set_title(f"N = {n}")
        ax.set_xlabel("predicted open count")
        ax.set_ylabel("true open count")
        ax.set_xticks(range(n + 1))
        ax.set_yticks(range(n + 1))
    fig.suptitle("Open count confusion, separated by channel count", fontsize=14)
    save(fig, "v5_open_confusion.png")


def state_scatter_figure(data: dict[str, np.ndarray], metrics: dict) -> None:
    truth = true_counts(data["r"])
    predicted = data["counts_pred"]
    rng = np.random.default_rng(4)
    sample = rng.choice(truth.shape[0] * truth.shape[2], 25_000, replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for s, ax in enumerate(axes.flat[:7]):
        t = truth[:, s].reshape(-1)[sample]
        p = predicted[:, s].reshape(-1)[sample]
        ax.hexbin(t, p, gridsize=35, mincnt=1, cmap="viridis", bins="log")
        limit = max(t.max(), p.max(), 1)
        ax.plot([0, limit], [0, limit], "k--", linewidth=1)
        row = metrics["by_state"][STATE_LETTERS[s]]
        ax.set_title(
            f"{STATE_LETTERS[s]}: {STATE_NAMES[s]}\n"
            f"MAE {row['mae']:.3f}, exact {100 * row['rounded_accuracy']:.1f}%, "
            f"r {row['correlation']:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("true count")
        ax.set_ylabel("predicted count")
    axes.flat[-1].axis("off")
    fig.suptitle("Predicted count beside true count for each state", fontsize=14)
    save(fig, "v5_state_scatter.png")


def state_trace_figure(data: dict[str, np.ndarray]) -> None:
    candidates = np.flatnonzero((data["N_true"] == 3) & (data["N_pred"] == 3))
    truth_all = true_counts(data["r"])
    errors = np.abs(data["counts_pred"][candidates] - truth_all[candidates]).mean(axis=(1, 2))
    i = int(candidates[np.argsort(errors)[len(candidates) // 2]])
    start, stop = 250, 650
    t = np.arange(start, stop)
    fig, axes = plt.subplots(7, 1, figsize=(15, 13), sharex=True)
    for s, ax in enumerate(axes):
        ax.step(t, truth_all[i, s, start:stop], where="post", color=GREEN, linewidth=1.8, label="true")
        ax.step(
            t,
            data["counts_pred"][i, s, start:stop],
            where="post",
            color=RED,
            linewidth=1.2,
            label="predicted",
        )
        ax.set_ylabel(f"{STATE_LETTERS[s]}\n{STATE_NAMES[s]}", rotation=0, ha="right", va="center")
        ax.set_ylim(-0.15, 3.15)
        ax.set_yticks(range(4))
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time (samples at 100 Hz)")
    fig.suptitle("All seven state counts on the same three-channel trace", fontsize=14)
    save(fig, "v5_state_traces.png")


def by_n_figure(metrics: dict) -> None:
    rows = metrics["by_channel_count"]
    n = np.arange(1, 6)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    plots = [
        ("n_accuracy", "N accuracy", True),
        ("open_accuracy", "open count accuracy", True),
        ("open_mae", "open count MAE", False),
        ("state_mae", "state count MAE", False),
    ]
    for ax, (key, title, percent) in zip(axes.flat, plots):
        values = [rows[str(i)][key] for i in n]
        bars = ax.bar(n, values, color=BLUE if percent else PURPLE)
        ax.set_xticks(n)
        ax.set_xlabel("true number of channels")
        ax.set_title(title)
        ax.set_ylim(0, 1.05 if percent else max(values) * 1.25)
        for bar, value in zip(bars, values):
            label = f"{100 * value:.1f}%" if percent else f"{value:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, label, ha="center", fontsize=9)
    fig.suptitle("Performance separated by the true number of channels", fontsize=14)
    save(fig, "v5_by_n.png")


def noise_figure(metrics: dict) -> None:
    rows = metrics["noise"]
    labels = ["x1", "x2", "x4"]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fields = [
        ("n_accuracy", "N accuracy", True),
        ("open_accuracy", "open count accuracy", True),
        ("state_mae", "state count MAE", False),
    ]
    for ax, (key, title, percent) in zip(axes, fields):
        values = [rows[label][key] for label in labels]
        bars = ax.bar(x, values, color=[BLUE, PURPLE, RED])
        ax.set_xticks(x, [f"noise {label}" for label in labels])
        ax.set_title(title)
        ax.set_ylim(0, 1.05 if percent else max(values) * 1.25)
        for bar, value in zip(bars, values):
            label = f"{100 * value:.1f}%" if percent else f"{value:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, label, ha="center")
    fig.suptitle("Performance as noise increases", fontsize=14)
    save(fig, "v5_noise.png")


def transition_figure(metrics: dict) -> None:
    row = metrics["transitions"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    labels = ["near a transition", "steady section"]
    values = [row["near_transition_accuracy"], row["steady_accuracy"]]
    bars = axes[0].bar(labels, values, color=[RED, GREEN])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Where errors happen")
    axes[0].set_ylabel("open count accuracy")
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{100 * value:.1f}%", ha="center")

    dwell = row["by_dwell_length"]
    keys = list(dwell)
    values = [dwell[key]["accuracy"] for key in keys]
    axes[1].plot(keys, values, marker="o", color=BLUE, linewidth=2)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("dwell length in samples")
    axes[1].set_ylabel("open count accuracy")
    axes[1].set_title("Longer states are easier")
    for i, value in enumerate(values):
        axes[1].text(i, value + 0.03, f"{100 * value:.0f}%", ha="center", fontsize=8)

    errors = row["absolute_error_counts"]
    keys = [int(key) for key in errors]
    total = sum(errors.values())
    values = [errors[str(key)] / total for key in keys]
    axes[2].bar(keys, values, color=PURPLE)
    axes[2].set_yscale("log")
    axes[2].set_xticks(keys)
    axes[2].set_xlabel("absolute open count error")
    axes[2].set_ylabel("fraction of samples, log scale")
    axes[2].set_title("Almost every miss is off by one")
    for key, value in zip(keys, values):
        axes[2].text(key, value * 1.3, f"{100 * value:.3g}%", ha="center", fontsize=8)
    fig.suptitle("Open count errors and transitions", fontsize=14)
    save(fig, "v5_transition.png")


def n_results_figure(data: dict[str, np.ndarray], metrics: dict) -> None:
    cm = np.asarray(metrics["channel_count"]["confusion"])
    pred = data["N_pred"]
    truth = data["N_true"]
    sorted_evidence = np.sort(data["evidence"], axis=1)
    gap = sorted_evidence[:, -1] - sorted_evidence[:, -2]
    correct = pred == truth
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(cm, cmap="Blues")
    for row in range(5):
        for col in range(5):
            axes[0].text(col, row, cm[row, col], ha="center", va="center")
    axes[0].set_xticks(range(5), range(1, 6))
    axes[0].set_yticks(range(5), range(1, 6))
    axes[0].set_xlabel("predicted N")
    axes[0].set_ylabel("true N")
    axes[0].set_title("N confusion matrix")

    axes[1].boxplot([gap[correct], gap[~correct]], tick_labels=["right", "wrong"], showfliers=False)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("log evidence gap")
    axes[1].set_title("Wrong counts have weaker evidence")
    axes[1].text(
        1,
        np.median(gap[correct]),
        f"median {np.median(gap[correct]):.0f}",
        ha="center",
        va="bottom",
    )
    axes[1].text(
        2,
        np.median(gap[~correct]),
        f"median {np.median(gap[~correct]):.0f}",
        ha="center",
        va="bottom",
    )

    mask = np.isin(truth, [4, 5])
    pairs, pair_counts = np.unique(np.c_[truth[mask], pred[mask]], axis=0, return_counts=True)
    labels = [f"true {t}\npred {p}" for (t, p) in pairs]
    axes[2].bar(labels, pair_counts, color=[GREEN if t == p else RED for t, p in pairs])
    axes[2].set_ylabel("traces")
    axes[2].set_title("N = 4 and N = 5")
    axes[2].tick_params(axis="x", labelsize=8)
    fig.suptitle(f"Channel count accuracy: {100 * metrics['total']['n_accuracy']:.1f}%", fontsize=14)
    save(fig, "v5_n_results.png")


def rate_figure(data: dict[str, np.ndarray], metrics: dict) -> None:
    fisher = np.asarray(json.loads((RESULTS / "fisher_synth_v2.json").read_text())["fisher"])
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    basis = eigenvectors[:, np.argsort(eigenvalues)[::-1][:4]]
    log_pred, log_true = np.log(data["rates_pred"]), np.log(data["R_true"])
    eff_pred, eff_true = effective_params(data["rates_pred"]), effective_params(data["R_true"])
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    panels = [
        (log_true @ basis[:, i], log_pred @ basis[:, i], f"rate direction {i + 1}")
        for i in range(4)
    ]
    panels += [
        (eff_true[:, 0], eff_pred[:, 0], "effective opening rate"),
        (eff_true[:, 1], eff_pred[:, 1], "effective closing rate"),
        (eff_true[:, 2], eff_pred[:, 2], "open probability"),
    ]
    r2_values = metrics["rates"]["direction_r2"] + [
        metrics["rates"]["effective_opening_r2"],
        metrics["rates"]["effective_closing_r2"],
        metrics["rates"]["open_probability_r2"],
    ]
    for ax, (truth, pred, title), score in zip(axes.flat[:7], panels, r2_values):
        ax.scatter(truth, pred, s=18, alpha=0.75, color=BLUE)
        lo, hi = min(truth.min(), pred.min()), max(truth.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(f"{title}\nR² {score:.3f}", fontsize=10)
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
    axes.flat[-1].axis("off")
    fig.suptitle("Predicted rates beside true rates", fontsize=14)
    save(fig, "v5_rates.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    data, metrics = load_results()
    summary_figure(metrics)
    architecture_figure()
    for n in range(1, 6):
        example_figure(data, n)
    open_confusion_figure(data)
    state_scatter_figure(data, metrics)
    state_trace_figure(data)
    by_n_figure(metrics)
    noise_figure(metrics)
    transition_figure(metrics)
    n_results_figure(data, metrics)
    rate_figure(data, metrics)


if __name__ == "__main__":
    main()

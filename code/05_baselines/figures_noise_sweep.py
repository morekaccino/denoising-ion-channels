"""Line-plot comparison across the noise-factor sweep.

Reads ``results/noise_sweep_<method>.json`` written by ``noise_sweep.py`` and
renders three line charts (channel-count accuracy, open-count accuracy,
open-count MAE) with one line per method, plus a combined panel.

Usage:
  python code/05_baselines/figures_noise_sweep.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "code" / "05_baselines" / "results"
FIGS = ROOT / "code" / "05_baselines" / "figures"

METHODS = [
    ("v5", "KI-HMM v5a", "#111111", "-", "o", 2.2),
    ("sdmc", "SD-HMM", "#1f77b4", "-", "s", 1.4),
    ("vnd", "VND-HMM", "#ff7f0e", "-", "^", 1.4),
    ("idc", "IDC", "#2ca02c", "-", "D", 1.4),
    ("deepchannel", "Deep-Channel", "#d62728", "-", "v", 1.4),
]
PANELS = [
    ("n_acc", "Channel-count accuracy", "noise_sweep_n_accuracy.png", (0.0, 1.02)),
    ("open_acc", "Open-count accuracy", "noise_sweep_open_accuracy.png", (0.0, 1.02)),
    ("open_mae", "Open-count MAE", "noise_sweep_open_mae.png", None),
]


def load() -> dict:
    out = {}
    for tag, label, color, style, marker, width in METHODS:
        path = RESULTS / f"noise_sweep_{tag}.json"
        if not path.exists():
            print(f"missing {path.name}, skipping {label}", file=sys.stderr)
            continue
        out[tag] = json.loads(path.read_text())
    return out


def draw(ax, data: dict, key: str, ylim) -> None:
    for tag, label, color, style, marker, width in METHODS:
        if tag not in data:
            continue
        d = data[tag]
        ax.plot(d["levels"], d[key], style, color=color, marker=marker,
                markersize=5, linewidth=width, label=label)
    ax.set_xlabel("noise factor")
    ax.set_xticks(np.round(np.arange(1.0, 4.01, 0.5), 2))
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)


def main() -> None:
    data = load()
    if not data:
        raise SystemExit("no sweep results found")
    FIGS.mkdir(parents=True, exist_ok=True)
    for key, title, fname, ylim in PANELS:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        draw(ax, data, key, ylim)
        ax.set_ylabel(title)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGS / fname, dpi=150)
        plt.close(fig)
        print("saved", FIGS / fname)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (key, title, _, ylim) in zip(axes, PANELS):
        draw(ax, data, key, ylim)
        ax.set_ylabel(title)
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "noise_sweep_combined.png", dpi=150)
    plt.close(fig)
    print("saved", FIGS / "noise_sweep_combined.png")


if __name__ == "__main__":
    main()

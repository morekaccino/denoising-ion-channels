"""Run KI-HMM v5a on real recordings for presentation plots.

Real recordings do not have ground-truth channel counts or state labels.  This
script therefore makes no accuracy claim and does not show rate estimates.  It
shows only the model output: predicted N, open count and mean state occupancy.

Usage:
  python code/04_ml/presentation_real_v5.py
"""

from __future__ import annotations

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

import torch_models_v4 as V4  # noqa: E402
from apply_kihmm_v2_real import file_segments  # noqa: E402
from eval_kihmm_v4 import load_model  # noqa: E402
from source.moreka import AxonData  # noqa: E402

MODEL = ROOT / "code" / "04_ml" / "models" / "kihmm_v4_v5a.pt"
RESULTS = ROOT / "code" / "04_ml" / "results"
FIGURES = RESULTS / "figures"
OUT = RESULTS / "presentation_real_v5.json"
STATE_NAMES = ["C1a", "C1b", "C2", "O1", "O2", "C3", "C4"]
STATE_LETTERS = list("abcdefg")


def main() -> None:
    model = load_model(str(MODEL), "cpu")
    source = AxonData(dirname=str(ROOT / "data" / "raw"))
    rows = []
    examples = []
    for index, filename in enumerate(source.filenames):
        signal = source[index]["signal"].values[::100]
        segments, count = file_segments(signal)
        if count == 0:
            continue
        x = torch.as_tensor(segments, dtype=torch.float32)
        group = torch.zeros(len(x), dtype=torch.long)
        prediction = V4.predict(model, x, group, emission_chunk=250)
        n_hat = prediction["N_hat"].cpu().numpy()
        counts = prediction["counts"].cpu().numpy()
        e_open = prediction["e_open"].cpu().numpy()
        row = {
            "index": index,
            "file": pathlib.Path(filename).name,
            "segments": int(count),
            "n_mean": float(n_hat.mean()),
            "n_mode": int(np.bincount(n_hat).argmax()),
            "state_counts_mean": counts.mean(axis=(0, 2)).tolist(),
        }
        rows.append(row)
        examples.append(
            {
                "index": index,
                "file": row["file"],
                "X": segments[0],
                "open": e_open[0],
                "N": int(n_hat[0]),
            }
        )

    # Use low, middle and high model-predicted channel counts.  These are
    # examples of model output, not evidence that those counts are correct.
    order = np.argsort([row["n_mean"] for row in rows])
    selected = [examples[int(order[0])], examples[int(order[len(order) // 2])], examples[int(order[-1])]]

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for ax, example in zip(axes, selected):
        t = np.arange(len(example["X"]))
        ax.plot(t, example["X"], color="#94a3b8", linewidth=0.7, label="real recording")
        ax2 = ax.twinx()
        ax2.step(t, example["open"], where="post", color="#dc2626", linewidth=1.4,
                 label="predicted open count")
        ax2.set_ylabel("predicted open channels", color="#dc2626")
        ax.set_ylabel("recorded current")
        ax.set_title(f"{example['file']}, predicted N = {example['N']}")
        if ax is axes[0]:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    axes[-1].set_xlabel("time (samples at 100 Hz)")
    fig.suptitle("Model output on real recordings, with no ground-truth labels", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "v5_real_examples.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    modes = np.asarray([row["n_mode"] for row in rows])
    values, counts = np.unique(modes, return_counts=True)
    axes[0].bar(values, counts, color="#2563eb")
    axes[0].set_xticks(range(1, 6))
    axes[0].set_xlabel("predicted N")
    axes[0].set_ylabel("recording files")
    axes[0].set_title(f"Predicted channel count across {len(rows)} usable files")
    for x, value in zip(values, counts):
        axes[0].text(x, value + 0.4, str(value), ha="center")

    occupancy = np.asarray([row["state_counts_mean"] for row in rows])
    means = occupancy.mean(axis=0)
    lo, hi = np.quantile(occupancy, [0.25, 0.75], axis=0)
    axes[1].bar(
        range(7),
        means,
        yerr=np.vstack([means - lo, hi - means]),
        color="#7c3aed",
        capsize=4,
    )
    axes[1].set_xticks(
        range(7),
        [f"{letter}\n{name}" for letter, name in zip(STATE_LETTERS, STATE_NAMES)],
    )
    axes[1].set_ylabel("predicted channels in state")
    axes[1].set_title("Average predicted state occupancy, median half shown")
    fig.suptitle("Real recording outputs are not accuracy measurements", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "v5_real_summary.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "warning": (
                    "Real recordings have no true N or state labels. These values are "
                    "model output, not measured accuracy. Rate output is omitted because "
                    "it does not transfer reliably."
                ),
                "files": rows,
                "selected_examples": [
                    {"index": row["index"], "file": row["file"], "N": row["N"]}
                    for row in selected
                ],
            },
            indent=2,
        )
    )
    print(f"processed {len(rows)} files")
    print("predicted N modes:", dict(zip(values.tolist(), counts.tolist())))
    print("saved", FIGURES / "v5_real_examples.png")
    print("saved", FIGURES / "v5_real_summary.png")
    print("saved", OUT)


if __name__ == "__main__":
    main()

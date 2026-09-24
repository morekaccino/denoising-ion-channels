"""Aggregate baseline results and compare against KI-HMM v5 on synth_v2.

Reads the frozen KI-HMM v5 metrics (published presentation JSON) and the
baseline result JSONs written by ``vnd_port.py``, ``sdmc_port.py`` and
``deepchannel_port.py``; writes a paper-ready table and grouped bar figures.

Usage:
  python code/05_baselines/compare_baselines.py --dc-tag dc_long
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "code" / "05_baselines" / "results"
V5_DIR = ROOT / "code" / "04_ml" / "results"
FIGS = ROOT / "code" / "05_baselines" / "figures"

SPLITS = ("test_x1", "test_x2", "test_x4")
METHODS = {
    "sdmc": "SD-HMM",
    "vnd": "VND-HMM",
}
CAPABILITIES = {
    "KI-HMM v5a": {"N": "yes", "open count": "yes", "7-state counts": "yes",
                   "rates": "yes (4 identifiable dirs)", "neural": "hybrid", "CFTR": "yes"},
    "SD-HMM": {"N": "yes (BIC)", "open count": "yes (Viterbi)", "7-state counts": "no",
               "rates": "effective birth-death only", "neural": "no", "CFTR": "no"},
    "VND-HMM": {"N": "yes (BIC)", "open count": "yes (Viterbi)", "7-state counts": "no",
                "rates": "transition probs only", "neural": "no", "CFTR": "no"},
    "Deep-Channel": {"N": "max-openings heuristic", "open count": "yes (per sample)",
                     "7-state counts": "no", "rates": "no", "neural": "yes (RCNN)", "CFTR": "no"},
}


def v5_rows() -> dict:
    pres = json.loads((V5_DIR / "presentation_v5_metrics.json").read_text())
    rows = {}
    for split, key in (("test_x1", "x1"), ("test_x2", "x2"), ("test_x4", "x4")):
        n = pres["noise"][key]
        rows[split] = {"n_acc": n["n_accuracy"], "open_acc": n["open_accuracy"],
                       "open_mae": n["open_mae"], "state_mae": n["state_mae"],
                       "per_n": None}
    rows["test_x1"]["per_n"] = {
        str(int(k)): {"traces": v["traces"], "n_acc": v["n_accuracy"],
                      "open_acc": v["open_accuracy"], "open_mae": v["open_mae"]}
        for k, v in pres["by_channel_count"].items()}
    return rows


def baseline_rows(tag: str) -> dict:
    rows = {}
    for split in SPLITS:
        path = RESULTS / f"{tag}_{split}.json"
        if not path.exists():
            rows[split] = None
            continue
        m = json.loads(path.read_text())["metrics"]
        rows[split] = {"n_acc": m["n_acc"], "open_acc": m["open_acc"],
                       "open_mae": m["open_mae"], "per_n": m.get("per_n")}
    return rows


def markdown_table(all_rows: dict) -> str:
    lines = ["| Method | Split | N acc | Open acc | Open MAE |",
             "|---|---|---|---|---|"]
    order = ["KI-HMM v5a", "SD-HMM", "VND-HMM", "Deep-Channel"]
    for name in order:
        rows = all_rows[name]
        for split in SPLITS:
            r = rows.get(split)
            if r is None:
                lines.append(f"| {name} | {split} | pending | pending | pending |")
                continue
            lines.append(f"| {name} | {split} | {r['n_acc']:.3f} | "
                         f"{r['open_acc']:.4f} | {r['open_mae']:.4f} |")
    return "\n".join(lines)


def per_n_table(all_rows: dict) -> str:
    lines = ["| True N | Metric | KI-HMM v5a | SD-HMM | VND-HMM | Deep-Channel |",
             "|---|---|---|---|---|---|"]
    names = ["KI-HMM v5a", "SD-HMM", "VND-HMM", "Deep-Channel"]
    for n in range(1, 6):
        for metric in ("n_acc", "open_acc"):
            cells = []
            for name in names:
                r = (all_rows[name].get("test_x1") or {}).get("per_n")
                v = (r or {}).get(str(n), {}).get(metric)
                cells.append("pending" if v is None else f"{v:.3f}")
            label = "N acc" if metric == "n_acc" else "open acc"
            lines.append(f"| {n} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def capability_table() -> str:
    cols = ["N", "open count", "7-state counts", "rates", "neural", "CFTR"]
    lines = ["| Method | " + " | ".join(cols) + " |",
             "|---|" + "---|" * len(cols)]
    for name, caps in CAPABILITIES.items():
        lines.append(f"| {name} | " + " | ".join(caps[c] for c in cols) + " |")
    return "\n".join(lines)


def figure(all_rows: dict, path: pathlib.Path) -> None:
    names = ["KI-HMM v5a", "SD-HMM", "VND-HMM", "Deep-Channel"]
    metrics = [("n_acc", "channel-count accuracy"),
               ("open_acc", "open-count accuracy"),
               ("open_mae", "open-count MAE")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(SPLITS))
    width = 0.2
    for ax, (metric, title) in zip(axes, metrics):
        for i, name in enumerate(names):
            vals = []
            for split in SPLITS:
                r = all_rows[name].get(split)
                vals.append(np.nan if r is None else r[metric])
            ax.bar(x + (i - 1.5) * width, vals, width, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(["noise x1", "noise x2", "noise x4"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dc-tag", default="dc_long")
    args = parser.parse_args()

    all_rows = {"KI-HMM v5a": v5_rows()}
    for tag, name in METHODS.items():
        all_rows[name] = baseline_rows(tag)
    all_rows["Deep-Channel"] = baseline_rows(args.dc_tag)

    md = ["# Baseline comparison (frozen synth_v2)", "",
          "## Headline metrics", markdown_table(all_rows), "",
          "## Per-N breakdown (noise x1)", per_n_table(all_rows), "",
          "## Capability matrix", capability_table(), ""]
    (RESULTS / "baseline_comparison.md").write_text("\n".join(md))
    (RESULTS / "baseline_comparison.json").write_text(json.dumps(all_rows, indent=2))
    figure(all_rows, FIGS / "baseline_comparison.png")
    print("\n".join(md))
    print("saved", RESULTS / "baseline_comparison.md")


if __name__ == "__main__":
    main()

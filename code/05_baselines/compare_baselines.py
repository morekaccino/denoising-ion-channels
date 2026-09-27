"""Aggregate baseline results and compare against KI-HMM v5 on synth_v2.

Reads the frozen KI-HMM v5 metrics (published presentation JSON, or the
per-trace predictions written by ``v5_reference.py`` when available) and the
baseline result JSONs written by ``vnd_port.py``, ``sdmc_port.py``, ``idc_port.py``,
``deepchannel_port.py``, ``moffett_port.py`` (and ``albertsen_port.py`` when
present); writes a paper-ready table and grouped bar figure.

Usage:
  python code/05_baselines/compare_baselines.py --dc-tag deepchannel
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
CAPABILITIES = {
    "KI-HMM v5a": {"N": "yes", "open count": "yes", "7-state counts": "yes",
                   "rates": "yes (4 identifiable dirs)", "learned emissions": "yes",
                   "CFTR": "yes"},
    "SD-HMM": {"N": "yes (BIC)", "open count": "yes (Viterbi)", "7-state counts": "no",
               "rates": "effective birth-death only", "learned emissions": "no",
               "CFTR": "no"},
    "VND-HMM": {"N": "yes (BIC)", "open count": "yes (Viterbi)", "7-state counts": "no",
                "rates": "transition probs only", "learned emissions": "no",
                "CFTR": "no"},
    "IDC": {"N": "yes (level count)", "open count": "yes (discretised levels)",
            "7-state counts": "no", "rates": "VND min-distance probs",
            "learned emissions": "no", "CFTR": "no"},
    "Deep-Channel": {"N": "max-openings heuristic", "open count": "yes (per sample)",
                     "7-state counts": "no", "rates": "no",
                     "learned emissions": "yes (RCNN)", "CFTR": "no"},
    "Moffett 2022": {"N": "no (single channel)", "open count": "N=1 only",
                     "7-state counts": "yes (single channel)",
                     "rates": "yes (per-trace EM)", "learned emissions": "no",
                     "CFTR": "yes"},
    "Albertsen 1994": {"N": "yes (likelihood)", "open count": "no",
                       "7-state counts": "no", "rates": "yes",
                       "learned emissions": "no", "CFTR": "no"},
}

OPEN_STATES = (3, 4)


def v5_rows() -> dict:
    """Per-trace v5 predictions when available, else the published aggregates."""
    rows = {}
    for split in SPLITS:
        cache = RESULTS / f"v5_{split}_predictions.npz"
        if cache.exists():
            with np.load(cache) as d:
                counts, y, N_pred, N_true = (d["counts"], d["y"], d["N_pred"], d["N_true"])
                open_expected = counts[:, OPEN_STATES].sum(axis=1)
                rows[split] = {
                    "n_acc": float((N_pred == N_true).mean()),
                    "open_acc": float((np.round(open_expected) == y).mean()),
                    "open_mae": float(np.abs(open_expected - y).mean()),
                    "state_mae": None,
                    "per_n": {str(k): {
                        "traces": int((N_true == k).sum()),
                        "n_acc": float((N_pred[N_true == k] == k).mean()),
                        "open_acc": float((np.round(open_expected[N_true == k]) == y[N_true == k]).mean()),
                        "open_mae": float(np.abs(open_expected[N_true == k] - y[N_true == k]).mean()),
                    } for k in range(1, 6) if (N_true == k).any()},
                }
        else:
            rows[split] = None
    if all(rows[s] is None for s in SPLITS):
        pres = json.loads((V5_DIR / "presentation_v5_metrics.json").read_text())
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


def baseline_rows(tag: str, n1_only: bool = False) -> dict:
    rows = {}
    for split in SPLITS:
        path = RESULTS / f"{tag}_{split}.json"
        if not path.exists():
            rows[split] = None
            continue
        m = json.loads(path.read_text())["metrics"]
        per_n = m.get("per_n")
        if per_n is None and n1_only:
            per_n = {"1": {"traces": None, "n_acc": m.get("n_acc"),
                           "open_acc": m.get("open_acc"),
                           "open_mae": m.get("open_mae")}}
        rows[split] = {"n_acc": m.get("n_acc"), "open_acc": m["open_acc"],
                       "open_mae": m["open_mae"], "per_n": per_n,
                       "state_acc": m.get("state_acc")}
    return rows


def all_baselines(dc_tag: str) -> dict:
    out = {
        "SD-HMM": baseline_rows("sdmc"),
        "VND-HMM": baseline_rows("vnd"),
        "IDC": baseline_rows("idc"),
        "Deep-Channel": baseline_rows(dc_tag),
        "Moffett 2022": baseline_rows("moffett", n1_only=True),
    }
    if (RESULTS / "albertsen_test_x1.json").exists():
        out["Albertsen 1994"] = baseline_rows("albertsen")
    return out


def is_n1(name: str) -> bool:
    return name == "Moffett 2022"


def markdown_table(all_rows: dict) -> str:
    lines = ["| Method | Split | N acc | Open acc | Open MAE | Notes |",
             "|---|---|---|---|---|---|"]
    for name, rows in all_rows.items():
        for split in SPLITS:
            r = rows.get(split)
            if r is None:
                lines.append(f"| {name} | {split} | pending | pending | pending | |")
                continue
            note = "N=1 traces only" if is_n1(name) else ""
            n_acc = "--" if r["n_acc"] is None else f"{r['n_acc']:.3f}"
            lines.append(f"| {name} | {split} | {n_acc} | "
                         f"{r['open_acc']:.4f} | {r['open_mae']:.4f} | {note} |")
    return "\n".join(lines)


def per_n_table(all_rows: dict) -> str:
    names = list(all_rows)
    lines = ["| True N | Metric | " + " | ".join(names) + " |",
             "|---|" + "---|" * (len(names) + 1)]
    for n in range(1, 6):
        for metric, label in (("n_acc", "N acc"), ("open_acc", "open acc")):
            cells = []
            for name in names:
                r = (all_rows[name].get("test_x1") or {}).get("per_n")
                v = (r or {}).get(str(n), {}).get(metric)
                cells.append("--" if v is None else f"{v:.3f}")
            lines.append(f"| {n} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def capability_table() -> str:
    cols = ["N", "open count", "7-state counts", "rates", "learned emissions", "CFTR"]
    lines = ["| Method | " + " | ".join(cols) + " |",
             "|---|" + "---|" * len(cols)]
    for name, caps in CAPABILITIES.items():
        lines.append(f"| {name} | " + " | ".join(caps[c] for c in cols) + " |")
    return "\n".join(lines)


def figure(all_rows: dict, path: pathlib.Path) -> None:
    names = list(all_rows)
    metrics = [("n_acc", "channel-count accuracy"),
               ("open_acc", "open-count accuracy"),
               ("open_mae", "open-count MAE")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(SPLITS))
    width = 0.8 / max(len(names), 1)
    for ax, (metric, title) in zip(axes, metrics):
        for i, name in enumerate(names):
            vals = []
            for split in SPLITS:
                r = all_rows[name].get(split)
                v = None if r is None else r.get(metric)
                vals.append(np.nan if v is None else v)
            ax.bar(x + (i - len(names) / 2 + 0.5) * width, vals, width, label=name)
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
    parser.add_argument("--dc-tag", default="deepchannel")
    args = parser.parse_args()

    all_rows = {"KI-HMM v5a": v5_rows()}
    all_rows.update(all_baselines(args.dc_tag))

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

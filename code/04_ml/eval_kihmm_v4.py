"""Evaluate KI-HMM v4 and place it on the leaderboard.

Reports, on the frozen synth_v2 splits: channel-count accuracy, per-state count
error, open-count accuracy, identifiable-direction R2 for the rates, the
bag-size ablation and the effective Markov parameters. The leaderboard pulls the
already-saved v2 and v3 numbers plus the Bayes-optimal ceiling from
``oracle_v4.py`` so every row is measured on the same traces.

Writes ``code/04_ml/results/kihmm_v4_eval.json`` and a summary figure.

Usage:
  python code/04_ml/eval_kihmm_v4.py --model code/04_ml/models/kihmm_v4_v4a.pt
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

import benchmark_v2 as B2  # noqa: E402
import torch_kinetics as TK  # noqa: E402
import torch_models_v4 as V4  # noqa: E402
from eval_kihmm_v2 import effective_params, r2  # noqa: E402
from train_kihmm_v4 import batch, evaluate, load  # noqa: E402
from train_kihmm_v2 import group_batches  # noqa: E402

RESULTS = ROOT / "code" / "04_ml" / "results"
FIGURES = RESULTS / "figures"
MISMATCH = ["mismatch_base", "mismatch_lowpass", "mismatch_gaussian",
            "mismatch_corr", "mismatch_drift"]


def load_model(path: str, device: str) -> V4.KIHMMv4:
    ckpt = torch.load(path, map_location=device)
    a = ckpt["args"]
    model = V4.KIHMMv4(hidden=a.get("hidden", 64), n_comp=a.get("n_comp", 4),
                       rate_stats=a.get("rate_stats", False),
                       n_head=a.get("n_head") or ("trace" if a.get("deep_n_head") else "pooled")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {path} (best epoch {ckpt.get('epoch')})", flush=True)
    return model


@torch.no_grad()
def predict_rates(model: V4.KIHMMv4, data: dict, device: str, per_batch: int = 8):
    preds = []
    for gids in group_batches(len(data["R"]), per_batch, np.random.default_rng(0), shuffle=False):
        b = batch(data, gids, device)
        preds.append(V4.predict(model, b["x"], b["group"], emission_chunk=250)["rates"].cpu().numpy())
    return np.log(np.concatenate(preds)), np.log(data["R"])


def leaderboard(v4: dict) -> dict:
    """Same traces, same metrics, every method that has been run on synth_v2."""
    def read(name, key):
        path = RESULTS / name
        return json.loads(path.read_text()) if path.exists() else None

    rows = {}
    for label, fname, path in (
        ("v2a (per-timestep head)", "kihmm_v2_v2a.json", "test"),
        ("v3a (neural HMM head)", "kihmm_v3_v3a.json", "test"),
    ):
        d = read(fname, path)
        if d:
            rows[label] = {f"x{s}": {"open_acc": d["test"][f"scale_{s}"].get("open_acc"),
                                     "state_mae": d["test"][f"scale_{s}"].get("state_mae"),
                                     "n_acc": d["test"][f"scale_{s}"].get("n_acc"),
                                     "dir_r2": d["test"][f"scale_{s}"].get("dir_r2")}
                           for s in ("1", "2", "4")}
    rows["v4 (this model)"] = {f"x{s}": {k: v4["splits"][f"test_x{s}"].get(k)
                                         for k in ("open_acc", "state_mae", "n_acc", "dir_r2")}
                               for s in ("1", "2", "4")}
    orc = read("oracle_v4.json", "splits")
    if orc:
        rows["oracle (true N + true rates)"] = {
            f"x{s}": {"open_acc": orc["splits"][f"test_x{s}"]["open_acc"],
                      "state_mae": orc["splits"][f"test_x{s}"]["state_mae"],
                      "n_acc": 1.0, "dir_r2": None}
            for s in ("1", "2", "4") if f"test_x{s}" in orc["splits"]}
    return rows


def summary_figure(report: dict, path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(19, 4))
    order = ["v2a (per-timestep head)", "v3a (neural HMM head)", "v4 (this model)",
             "oracle (true N + true rates)"]
    rows = [k for k in order if k in report["leaderboard"]]
    short = ["v2a", "v3a", "v4", "oracle"][: len(rows)]
    width = 0.25
    for j, s in enumerate(("1", "2", "4")):
        vals = [report["leaderboard"][k][f"x{s}"]["open_acc"] or 0 for k in rows]
        ax[0].bar(np.arange(len(rows)) + (j - 1) * width, vals, width, label=f"noise x{s}")
    ax[0].set_xticks(range(len(rows)), short)
    ax[0].set_ylabel("open-count accuracy")
    ax[0].set_title("open-count accuracy (frozen test)")
    ax[0].legend(fontsize=8)

    for j, s in enumerate(("1", "2", "4")):
        vals = [report["leaderboard"][k][f"x{s}"]["state_mae"] or 0 for k in rows]
        ax[1].bar(np.arange(len(rows)) + (j - 1) * width, vals, width, label=f"noise x{s}")
    ax[1].set_xticks(range(len(rows)), short)
    ax[1].set_ylabel("per-state count MAE (channels)")
    ax[1].set_title("a..g error (lower is better)")

    r2s = report["splits"]["test_x1"]["rate_r2_per_rate"]
    ax[2].bar(range(len(r2s)), r2s)
    ax[2].set_xticks(range(len(r2s)), B2.RATE_NAMES, rotation=60, fontsize=7)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_title("per-rate R2 (test x1)")

    ks = sorted(report["bag_ablation_test_x1"], key=lambda s: int(s.split("=")[1]))
    ax[3].plot([int(k.split("=")[1]) for k in ks],
               [report["bag_ablation_test_x1"][k]["dir_r2"][0] for k in ks], "o-")
    ax[3].axhline(0, color="k", lw=0.5)
    ax[3].set_xlabel("traces per group")
    ax[3].set_ylabel("R2, top identifiable direction")
    ax[3].set_title("rate accuracy vs group size")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print("saved", path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "code" / "04_ml" / "models" / "kihmm_v4_v4a.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tag", default="kihmm_v4_eval")
    parser.add_argument("--skip-mismatch", action="store_true")
    args = parser.parse_args()

    model = load_model(args.model, args.device)
    F = np.asarray(json.loads((RESULTS / "fisher_synth_v2.json").read_text())["fisher"])
    evals, evecs = np.linalg.eigh(F)
    order = np.argsort(evals)[::-1]
    basis = evecs[:, order[:4]]

    splits = {"val": load("val")}
    splits.update({f"test_x{s}": load("test", s) for s in ("1", "2", "4")})
    splits.update({f"extrap_x{s}": load("extrap", s) for s in ("1", "2")})
    if not args.skip_mismatch:
        splits.update({m: load(m) for m in MISMATCH})

    report = {"model": str(args.model), "fisher_eigenvalues": np.round(evals[order], 4).tolist(),
              "splits": {}, "bag_ablation_test_x1": {}, "effective_rates": {}}
    for name, data in splits.items():
        m = evaluate(model, data, args.device, basis=basis)
        report["splits"][name] = m
        print(f"{name:20s} N={m['n_acc']:.3f} open={m['open_acc']:.4f} "
              f"state_mae={m['state_mae']:.4f} dirR2={np.round(m['dir_r2'], 3)}", flush=True)

    for k in range(1, splits["test_x1"]["K"] + 1):
        m = evaluate(model, splits["test_x1"], args.device, basis=basis, k_use=k)
        report["bag_ablation_test_x1"][f"K={k}"] = {
            "dir_r2": m["dir_r2"], "n_acc": m["n_acc"], "state_mae": m["state_mae"]}
        print(f"bag K={k}: dirR2={np.round(m['dir_r2'], 3)} N={m['n_acc']:.3f}", flush=True)

    for name in ("val", "test_x1"):
        pred, truth = predict_rates(model, splits[name], args.device)
        et, ep = effective_params(np.exp(truth)), effective_params(np.exp(pred))
        report["effective_rates"][name] = {
            "r2_log_opening": r2(ep[:, 0], et[:, 0]),
            "r2_log_closing": r2(ep[:, 1], et[:, 1]),
            "r2_p_open": r2(ep[:, 2], et[:, 2]),
            "median_rel_err_opening": float(np.median(np.abs(np.exp(ep[:, 0] - et[:, 0]) - 1))),
            "median_rel_err_closing": float(np.median(np.abs(np.exp(ep[:, 1] - et[:, 1]) - 1))),
        }
        print(f"effective {name}: { {k: round(v, 4) for k, v in report['effective_rates'][name].items()} }", flush=True)

    report["leaderboard"] = leaderboard(report)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.tag}.json").write_text(json.dumps(report, indent=2))
    try:
        summary_figure(report, FIGURES / f"{args.tag}.png")
    except Exception as exc:
        print("figure skipped:", exc, flush=True)
    print("saved", RESULTS / f"{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()

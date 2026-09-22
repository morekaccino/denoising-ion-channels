"""Evaluate KI-HMM v2: N, per-state counts, Markov parameters, bag-size ablation.

Writes ``code/04_ml/results/kihmm_v2_eval.json`` and figures under
``code/04_ml/results/figures/``.

Usage:
  python code/04_ml/eval_kihmm_v2.py --model code/04_ml/models/kihmm_v2_v2a.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from scipy.linalg import expm

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "04_ml"))

import benchmark as B  # noqa: E402
import benchmark_v2 as B2  # noqa: E402
import torch_models as TM  # noqa: E402
import torch_models_v2 as T  # noqa: E402
from train_kihmm_v2 import evaluate, load  # noqa: E402

RESULTS = ROOT / "code" / "04_ml" / "results"
FIGURES = RESULTS / "figures"


def effective_params(R_values: np.ndarray) -> np.ndarray:
    """(G,12) rates -> (G,3): log opening rate, log closing rate, p_open."""
    out = np.zeros((len(R_values), 3))
    for i, vals in enumerate(R_values):
        Pk, pik = TM.open_count_chain(1, expm(B.DT * B2.rate_table(vals)))
        out[i] = [np.log(Pk[1, 0] / B.DT), np.log(Pk[0, 1] / B.DT), float(pik[1])]
    return out


def predict_rates(model, data, per_batch: int = 16):
    from train_kihmm_v2 import batch_tensors, group_batches

    X, y, r, N, group, R, K = data
    model.eval()
    preds = []
    with torch.no_grad():
        for gids in group_batches(len(R), per_batch, np.random.default_rng(0), shuffle=False):
            xb, rb, Nb, Rb, gb = batch_tensors(X, r, N, R, K, gids)
            out = model(xb, gb)
            preds.append(out["rates_group"].cpu().numpy())
    return np.log(np.concatenate(preds)), np.log(R)


def r2(pred, truth):
    return float(1 - ((pred - truth) ** 2).mean() / max(truth.var(), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "code" / "04_ml" / "models" / "kihmm_v2_v2a.pt"))
    parser.add_argument("--tag", default="kihmm_v2_eval")
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location=T.DEVICE)
    model = T.KIHMMv2(hidden=ckpt["args"].get("hidden", 64)).to(T.DEVICE)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.model} (epoch {ckpt.get('epoch')})", flush=True)

    F = np.asarray(json.loads((RESULTS / "fisher_synth_v2.json").read_text())["fisher"])
    evals, evecs = np.linalg.eigh(F)
    order = np.argsort(evals)[::-1]
    basis = evecs[:, order[:4]]
    report = {
        "model": str(args.model),
        "fisher_eigenvalues": np.round(evals[order], 4).tolist(),
        "splits": {},
        "bag_ablation_test_x1": {},
        "effective_rates": {},
    }

    splits = {"val": load("val")}
    test = load("test")
    for s in ["1", "2", "4"]:
        d = B2.load_split("test")
        splits[f"test_x{s}"] = (d[f"X_s{s}"], test[1], test[2], test[3], test[4], test[5], test[6])
    extrap = load("extrap")
    for s in ["1", "2"]:
        d = B2.load_split("extrap")
        splits[f"extrap_x{s}"] = (d[f"X_s{s}"], extrap[1], extrap[2], extrap[3], extrap[4], extrap[5], extrap[6])

    for name, data in splits.items():
        m = evaluate(model, data, basis=basis)
        report["splits"][name] = m
        print(f"{name}: N={m['n_acc']:.3f} state_mae={m['state_mae']:.3f} "
              f"dirR2={m['dir_r2']} rate_mae={m['rate_mae']:.3f}", flush=True)

    for k in range(1, 7):
        m = evaluate(model, splits["test_x1"], basis=basis, k_use=k)
        report["bag_ablation_test_x1"][f"K={k}"] = {"dir_r2": m["dir_r2"], "n_acc": m["n_acc"],
                                                    "state_mae": m["state_mae"]}
        print(f"bag K={k}: dirR2={m['dir_r2']} N={m['n_acc']:.3f}", flush=True)

    for name in ["val", "test_x1"]:
        pred, truth = predict_rates(model, splits[name])
        eff_t, eff_p = effective_params(np.exp(truth)), effective_params(np.exp(pred))
        report["effective_rates"][name] = {
            "r2_log_opening": r2(eff_p[:, 0], eff_t[:, 0]),
            "r2_log_closing": r2(eff_p[:, 1], eff_t[:, 1]),
            "r2_p_open": r2(eff_p[:, 2], eff_t[:, 2]),
            "median_rel_err_opening": float(np.median(np.abs(np.exp(eff_p[:, 0] - eff_t[:, 0]) - 1))),
            "median_rel_err_closing": float(np.median(np.abs(np.exp(eff_p[:, 1] - eff_t[:, 1]) - 1))),
        }
        print(f"effective {name}: {report['effective_rates'][name]}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.tag}.json").write_text(json.dumps(report, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FIGURES.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        r2s = report["splits"]["test_x1"]["rate_r2_per_rate"]
        axes[0].bar(range(len(r2s)), r2s)
        axes[0].set_xticks(range(len(r2s)), B2.RATE_NAMES, rotation=60, fontsize=7)
        axes[0].set_title("per-rate R2 (test x1)")
        axes[0].axhline(0, color="k", lw=0.5)

        axes[1].bar(range(4), report["splits"]["test_x1"]["dir_r2"])
        axes[1].set_xticks(range(4), [f"dir{i+1}" for i in range(4)])
        axes[1].set_title("identifiable-direction R2 (test x1)")
        axes[1].axhline(0, color="k", lw=0.5)

        for k in range(1, 7):
            axes[2].plot(k, report["bag_ablation_test_x1"][f"K={k}"]["dir_r2"][0], "o-")
        axes[2].set_xlabel("traces per group")
        axes[2].set_ylabel("R2 dir1")
        axes[2].set_title("rate accuracy vs group size")
        fig.tight_layout()
        fig.savefig(FIGURES / f"{args.tag}.png", dpi=130)
        print("saved", FIGURES / f"{args.tag}.png", flush=True)
    except Exception as exc:
        print("figure skipped:", exc, flush=True)

    print("saved", RESULTS / f"{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()

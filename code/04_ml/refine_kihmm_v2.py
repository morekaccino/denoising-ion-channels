"""Hybrid KI-HMM v2 decoding: neural N + rates, exact kinetic-chain posterior.

The trained network proposes the channel count N and the rate table for a group
of traces. The exact factorial count chain (``kinetics.py``) with the known
generalized-hyperbolic emission model then performs forward-backward and returns
the posterior expected counts per kinetic state (a..g) for every time step.

Usage:
  python code/04_ml/refine_kihmm_v2.py --model code/04_ml/models/kihmm_v2_v2a.pt
  python code/04_ml/refine_kihmm_v2.py --splits test_x1 --oracle
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code" / "04_ml"))

import benchmark as B  # noqa: E402
import benchmark_v2 as B2  # noqa: E402
import kinetics as K  # noqa: E402
import torch_models_v2 as T  # noqa: E402
from eval_kihmm_v2 import predict_rates  # noqa: E402
from train_kihmm_v2 import batch_tensors, group_batches  # noqa: E402

RESULTS = ROOT / "code" / "04_ml" / "results"
FIGURES = RESULTS / "figures"


def load_model(path: str):
    ckpt = torch.load(path, map_location=T.DEVICE)
    model = T.KIHMMv2(hidden=ckpt["args"].get("hidden", 64)).to(T.DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _emission_cache(cache, N, scale):
    key = (N, round(float(scale), 3))
    if key not in cache:
        grid = K.default_grid(N, noise_scale=scale)
        cache[key] = (grid, K.emission_log_densities(N, grid, noise_scale=scale))
    return cache[key]


def decode_trace(X, N, rate_values, scale, cache):
    """Exact posterior expected state counts (7,T) and log evidence."""
    grid, logE = _emission_cache(cache, N, scale)
    P0 = expm(B.DT * B2.rate_table(np.asarray(rate_values, dtype=float)))
    P = K.count_transition_matrix(N, P0)
    pi = K.count_stationary(N, P0)
    states = K.count_states(N).astype(float)
    logB = K.log_emission_matrix(np.asarray(X, dtype=float), grid, logE, states)
    _, _, _, gamma, logZ = K.forward_backward(logB, P, pi)
    return (gamma @ states).T, float(logZ)


def refine_data(model, data, scale: float = 1.0, oracle: bool = False,
                evidence_n: bool = False, n_values=(1, 2, 3, 4, 5), per_batch: int = 8):
    """Model predictions plus exact-chain refinement for every trace."""
    X, y, r, N, group, R, Kt = data
    n_traces = len(X)
    T_steps = X.shape[1]
    pred_log, _ = predict_rates(model, data, per_batch=16)
    R_hat = np.exp(pred_log)

    counts_model = np.empty((n_traces, 7, T_steps), dtype=np.float32)
    counts_refined = np.empty((n_traces, 7, T_steps), dtype=np.float32)
    counts_oracle = np.empty_like(counts_refined) if oracle else None
    n_hat = np.empty(n_traces, dtype=int)
    n_evid = np.empty(n_traces, dtype=int) if evidence_n else None
    cache = {}
    for gids in group_batches(len(R), per_batch, np.random.default_rng(0), shuffle=False):
        idx = (np.asarray(gids)[:, None] * Kt + np.arange(Kt)[None, :]).reshape(-1)
        xb, rb, Nb, Rb, gb = batch_tensors(X, r, N, R, Kt, gids)
        with torch.no_grad():
            p = T.predict(model, xb, gb)
        counts_model[idx] = p["counts"].cpu().numpy()
        n_hat[idx] = p["N_hat"].cpu().numpy()
        for i in idx:
            gi = int(group[i])
            n = max(1, int(n_hat[i]))
            counts_refined[i], _ = decode_trace(X[i], n, R_hat[gi], scale, cache)
            if oracle:
                counts_oracle[i], _ = decode_trace(X[i], int(N[i]), R[gi], scale, cache)
            if evidence_n:
                best, best_z = n, -np.inf
                for cand in n_values:
                    _, lz = decode_trace(X[i], cand, R_hat[gi], scale, cache)
                    if lz > best_z:
                        best, best_z = cand, lz
                n_evid[i] = best
    out = {
        "N_hat": n_hat,
        "counts_model": counts_model,
        "counts_refined": counts_refined,
        "R_hat": R_hat,
    }
    if oracle:
        out["counts_oracle"] = counts_oracle
    if evidence_n:
        out["N_evidence"] = n_evid
    return out


def metrics(counts, y, r, N, N_pred=None):
    open_idx = np.flatnonzero(B.STATEMAP == 1)
    counts_true = np.stack([(r == s).sum(axis=1) for s in range(7)], axis=1).astype(float)
    mae = float(np.abs(counts - counts_true).mean())
    open_acc = float((np.round(counts[:, open_idx].sum(axis=1)) == y).mean())
    out = {"state_mae": mae, "open_acc": open_acc}
    if N_pred is not None:
        out["n_acc"] = float((N_pred == N).mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "code" / "04_ml" / "models" / "kihmm_v2_v2a.pt"))
    parser.add_argument("--splits", default="val,test_x1,test_x2,test_x4",
                        help="comma list of val/test_x1/test_x2/test_x4/extrap_x1/extrap_x2")
    parser.add_argument("--oracle", action="store_true", help="also decode with true N and rates")
    parser.add_argument("--evidence-n", action="store_true", help="N by exact marginal likelihood")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N traces (0 = all)")
    parser.add_argument("--tag", default="kihmm_v2_refined")
    args = parser.parse_args()

    model = load_model(args.model)
    from train_kihmm_v2 import load  # noqa: E402

    def split_data(name):
        if name.startswith("test_x"):
            s = name.replace("test_x", "")
            d = B2.load_split("test")
            base = load("test")
            return (d[f"X_s{s}"], base[1], base[2], base[3], base[4], base[5], base[6]), float(s)
        if name.startswith("extrap_x"):
            s = name.replace("extrap_x", "")
            d = B2.load_split("extrap")
            base = load("extrap")
            return (d[f"X_s{s}"], base[1], base[2], base[3], base[4], base[5], base[6]), float(s)
        return load(name), 1.0

    report = {}
    for name in args.splits.split(","):
        name = name.strip()
        data, scale = split_data(name)
        if args.limit:
            vals = list(data)
            K = int(vals[6])
            lim = max(K, (min(args.limit, len(vals[0])) // K) * K)
            for j in range(6):
                vals[j] = vals[j][:lim]
            vals[5] = vals[5][: lim // K]
            data = tuple(vals)
        res = refine_data(model, data, scale=scale, oracle=args.oracle, evidence_n=args.evidence_n)
        y, r, N = data[1], data[2], data[3]
        row = {
            "model": metrics(res["counts_model"], y, r, N, res["N_hat"]),
            "refined": metrics(res["counts_refined"], y, r, N, res["N_hat"]),
        }
        if args.oracle:
            row["oracle"] = metrics(res["counts_oracle"], y, r, N, N)
        if args.evidence_n:
            row["evidence_n"] = metrics(res["counts_model"], y, r, N, res["N_evidence"])
        report[name] = row
        print(f"{name:9s} model  open={row['model']['open_acc']:.3f} mae={row['model']['state_mae']:.3f} "
              f"N={row['model']['n_acc']:.3f} | refined open={row['refined']['open_acc']:.3f} "
              f"mae={row['refined']['state_mae']:.3f}"
              + (f" | oracle open={row['oracle']['open_acc']:.3f} mae={row['oracle']['state_mae']:.3f}"
                 if args.oracle else "")
              + (f" | evidN={row['evidence_n']['n_acc']:.3f}" if args.evidence_n else ""), flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.tag}.json").write_text(json.dumps(report, indent=2))
    print("saved", RESULTS / f"{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()

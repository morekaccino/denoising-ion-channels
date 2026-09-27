"""Controlled experiments for the fairness question: hidden assumptions.

2x2 design on the frozen test set (same N distribution, groups and rate tables):

                     generalized-hyperbolic noise | Gaussian noise
  7-state CFTR        frozen benchmark (original)  | emission test
  two-state channels  channel-structure test       | native control

- Gaussian cells rebuild the traces with per-channel Gaussian noise of the same
  state-dependent standard deviation measured on the frozen x1 traces
  (closed 0.128, open 0.232), keeping labels, groups and rates identical.
- Two-state cells replace the 7-state channels with the two-state projection of
  each group's CFTR rate table: effective opening/closing rates are the
  probability fluxes across the open/closed boundary of the stationary CFTR
  chain, so the stationary open probability and the boundary flux are the same.
  Everything else (N per trace, trace length, levels 0.58/1.4, noise model,
  groups) is unchanged.
- KI-HMM v5a is run on the two CFTR cells only; its model is 7-state by
  construction, so the two-state cells test the baselines under their own
  structural assumption.

Usage:
  python code/05_baselines/assumption_controls.py --cell all --method all --jobs 10
  python code/05_baselines/assumption_controls.py --cell twostate_gh --method sdmc
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import expm

BASEDIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASEDIR))
sys.path.insert(0, str(BASEDIR.parents[1] / "code" / "04_ml"))

import benchmark as B  # noqa: E402
import benchmark_v2 as B2  # noqa: E402
import hmm_core as core  # noqa: E402
import noise_sweep as NS  # noqa: E402

RESULTS = BASEDIR / "results"
SIGMA_CLOSED, SIGMA_OPEN = 0.1280, 0.2317
LEVEL_CLOSED, LEVEL_OPEN = 0.58, 1.4
OPEN_STATES = (3, 4)
SEED = 20260926


def cftr_data(gaussian: bool) -> dict:
    d = dict(np.load(core.SYNTH / "test.npz"))
    N, y, r = d["N"], d["y"], d["r"]
    level = LEVEL_CLOSED * N[:, None] + (LEVEL_OPEN - LEVEL_CLOSED) * y
    if gaussian:
        rng = np.random.default_rng(SEED)
        open_mask = np.isin(r, OPEN_STATES)
        sigma = np.where(open_mask, SIGMA_OPEN, SIGMA_CLOSED)
        valid = (r >= 0).astype(np.float64)
        X = (level + (rng.normal(0.0, sigma) * valid).sum(axis=1)).astype(np.float32)
    else:
        X = d["X_s1"]
    return {"X": X.astype(np.float64), "y": y, "r": r, "N": N,
            "group": d["group"], "R": d["R"],
            "log_scale": np.zeros(len(X), dtype=np.float32),
            "K": int((d["group"] == 0).sum())}


def projected_two_state(group_rates: np.ndarray) -> np.ndarray:
    """Two-state transition matrix with the same stationary open probability and
    open/closed boundary flux as the 7-state CFTR chain of the group."""
    Q = B2.rate_table(group_rates)
    pi = B.stationary(expm(B.DT * Q))
    closed, opn = B.STATEMAP == 0, B.STATEMAP == 1
    k_on = float((pi[closed, None] * Q[np.ix_(closed, opn)]).sum() / pi[closed].sum())
    k_off = float((pi[opn, None] * Q[np.ix_(opn, closed)]).sum() / pi[opn].sum())
    Q2 = np.array([[-k_on, k_on], [k_off, -k_off]])
    return expm(B.DT * Q2)


def twostate_data(gaussian: bool) -> dict:
    d = dict(np.load(core.SYNTH / "test.npz"))
    N, group, R = d["N"], d["group"], d["R"]
    n, T = len(N), d["y"].shape[1]
    rng = np.random.default_rng(SEED)
    P2 = {g: projected_two_state(R[g]) for g in np.unique(group)}
    y = np.zeros((n, T), dtype=np.int8)
    X = np.zeros((n, T), dtype=np.float64)
    for i in range(n):
        P = P2[int(group[i])]
        states = np.empty((N[i], T), dtype=np.int8)
        for c in range(N[i]):
            s = rng.choice(2, p=B.stationary(P))
            for t in range(T):
                states[c, t] = s
                s = rng.choice(2, p=P[s])
        y[i] = states.sum(axis=0)
        level = np.where(states == 1, LEVEL_OPEN, LEVEL_CLOSED)
        if gaussian:
            sigma = np.where(states == 1, SIGMA_OPEN, SIGMA_CLOSED)
            X[i] = (level + rng.normal(0.0, sigma)).sum(axis=0)
        else:
            chan = B._channel_observation(states, rng, noise_kind="gh", ar_rho=0.0)
            X[i] = (level + (chan - level)).sum(axis=0)
    return {"X": X.astype(np.float32).astype(np.float64), "y": y, "N": N,
            "group": group, "R": R,
            "log_scale": np.zeros(n, dtype=np.float32),
            "K": int((group == 0).sum())}


def run_moffett(data: dict) -> dict:
    import moffett_port as MP
    keep = np.flatnonzero(data["N"] == 1)
    paths = np.stack([MP.fit_trace(data["X"][i])["path"] for i in keep])
    open_hat = np.isin(paths, OPEN_STATES)
    open_true = data["y"][keep] > 0
    return {"n_acc": None,
            "open_acc": float((open_hat == open_true).mean()),
            "open_mae": float(np.abs(open_hat.astype(np.float64)
                                     - open_true.astype(np.float64)).mean())}


def run(method: str, data: dict, jobs: int) -> dict:
    if method == "v5":
        return NS.run_v5_data(data)
    if method in ("sdmc", "vnd"):
        return NS.run_hmm(method, data["X"], data["y"], data["N"], jobs)
    if method == "idc":
        return NS.run_idc(data["X"], data["y"], data["N"], jobs)
    if method == "deepchannel":
        return NS.metrics_from_open(NS.run_deepchannel(data["X"]),
                                    data["y"], data["N"])
    if method == "moffett":
        return run_moffett(data)
    raise ValueError(method)


CELLS = {
    "cftr_gh": lambda: cftr_data(gaussian=False),
    "cftr_gauss": lambda: cftr_data(gaussian=True),
    "twostate_gh": lambda: twostate_data(gaussian=False),
    "twostate_gauss": lambda: twostate_data(gaussian=True),
}
METHODS = ("v5", "sdmc", "vnd", "idc", "deepchannel", "moffett")
SKIP = {"twostate_gh": ("v5",), "twostate_gauss": ("v5",)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default="all",
                        choices=list(CELLS) + ["all"])
    parser.add_argument("--method", default="all", choices=list(METHODS) + ["all"])
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()
    cells = list(CELLS) if args.cell == "all" else [args.cell]
    methods = list(METHODS) if args.method == "all" else [args.method]
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "assumption_controls.json"
    report = json.loads(out_path.read_text()) if out_path.exists() else {"cells": {}}
    for cell in cells:
        data = CELLS[cell]()
        open_mean = float(data["y"].mean())
        print(f"=== {cell}: N mean {data['N'].mean():.2f}, open mean {open_mean:.3f}",
              flush=True)
        cell_out = report["cells"].setdefault(cell, {"seed": SEED,
                                                     "open_mean": open_mean,
                                                     "methods": {}})
        for method in methods:
            if method in SKIP.get(cell, ()):
                print(f"  {method}: skipped (7-state model)", flush=True)
                continue
            import time
            t0 = time.time()
            m = run(method, data, args.jobs)
            m["runtime_s"] = round(time.time() - t0, 1)
            cell_out["methods"][method] = m
            print(f"  {method}: open {m['open_acc']:.4f} MAE {m['open_mae']:.4f} "
                  f"({m['runtime_s']}s)", flush=True)
            out_path.write_text(json.dumps(report, indent=2))
    out_path.write_text(json.dumps(report, indent=2))
    print("saved", out_path)


if __name__ == "__main__":
    main()

"""Python port of the VND-HMM (Vanegas et al. 2024) with BIC model selection.

Reference implementation: https://github.com/ljvanegas/VND (GPL-2), files
``src/vnd_model.cpp``, ``R/estimate_HMM.R`` (``HMM_normal``/``HMM_custom``),
``src/Baum_Welch_step.cpp``, ``src/Likelihood_Forward.cpp``,
``src/Viterbi_simple.cpp``.

The model is a discrete-time HMM on the sum process ``S in {0..l}`` of ``l``
two-state channels. Parameters ``theta = (lambda_0, eta_1, ..., lambda_{l-1},
eta_l)`` where ``lambda_r`` is the probability that a closed channel stays
closed and ``eta_r`` the probability that an open channel stays open, given
``r`` channels open. Emissions are Gaussian with equidistant means and a
per-level standard deviation.

Usage:
  python code/05_baselines/vnd_port.py --verify
  python code/05_baselines/vnd_port.py --split test_x1 --limit 16
  python code/05_baselines/vnd_port.py --split test_x1 --jobs 10
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from math import comb

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hmm_core as core  # noqa: E402

RESULTS = core.ROOT / "code" / "05_baselines" / "results"
LS_DEFAULT = (1, 2, 3, 4, 5)


def vnd_transition(theta: np.ndarray, n_channels: int) -> np.ndarray:
    """Port of ``vnd_model`` (src/vnd_model.cpp)."""
    theta = np.asarray(theta, dtype=np.float64)
    l = n_channels
    p_stay_open = np.concatenate([[1.0], theta[1::2]])   # eta_0=1, eta_1..eta_l
    p_stay_closed = np.concatenate([theta[0::2], [1.0]])  # lambda_0..lambda_{l-1}, 1
    T = np.zeros((l + 1, l + 1))
    for i in range(l + 1):
        for j in range(l + 1):
            acc = 0.0
            for k in range(max(0, i - j), min(i, l - j) + 1):
                acc += (comb(i, k) * comb(l - i, j - i + k)
                        * p_stay_open[i] ** (i - k) * (1 - p_stay_open[i]) ** k
                        * p_stay_closed[i] ** (l - j - k) * (1 - p_stay_closed[i]) ** (j - i + k))
            T[i, j] = acc
    return T


def theta_m_step(xi: np.ndarray, n_channels: int, theta0: np.ndarray) -> np.ndarray:
    """Port of the transition M-step (``matrix_likelihood``/``log_cost``):
    maximise ``sum xi_ij log Q_ij(theta)`` over ``theta in [0, 1]``."""
    target = np.asarray(xi).ravel()
    n_par = 2 * n_channels

    def cost(x):
        q = np.clip(vnd_transition(x, n_channels), 1e-10, None)
        return float(-np.sum(target * np.log(q.ravel())))

    res = minimize(cost, np.asarray(theta0, dtype=np.float64), method="SLSQP",
                   bounds=[(0.0, 1.0)] * n_par, options={"maxiter": 300, "ftol": 1e-10})
    return res.x


def fit_trace(data: np.ndarray, n_channels: int, emit_range: tuple[float, float],
              max_it: int = 100, threshold: float = 1e-6) -> dict:
    """Port of ``HMM_normal``/``HMM_custom`` for one fixed channel count."""
    n_levels = n_channels + 1
    step = (emit_range[1] - emit_range[0]) / (n_levels - 1)
    if not np.isfinite(step) or step <= 0:
        step = 1.0
    emit = np.concatenate([[emit_range[0], step], np.full(n_levels, 10.0 / step)])
    theta = np.full(2 * n_channels, 0.9)
    P = vnd_transition(theta, n_channels)
    bw = None
    for _ in range(max_it):
        probs = core.gaussian_level_probabilities(data, emit, n_levels)
        bw = core.bw_step(data, probs, P, init=None)
        emit_new = core.normal_params(bw["gamma"], bw["y_gamma"], bw["y_gamma2"],
                                      n_levels, variant="vnd")
        theta_new = theta_m_step(bw["xi"], n_channels, theta)
        P_new = vnd_transition(theta_new, n_channels)
        err = max(np.abs(P_new - P).max(), np.abs(emit_new - emit).max())
        emit, theta, P = emit_new, theta_new, P_new
        if err < threshold:
            break

    mu = emit[0] + np.arange(n_levels) * emit[1]
    sigma = 1.0 / np.abs(emit[2:])
    ll = core.forward_loglik(data, bw["pi"], core.gaussian_densities(data, mu, sigma), P)
    path = core.viterbi(data, bw["pi"], P, mu, sigma**2)
    n_pars = 3 * n_channels + 3
    n_obs = len(data)
    return {
        "l": n_channels,
        "loglik": float(ll),
        "k": n_pars,
        "bic": float(n_pars * np.log(n_obs) - 2 * ll),
        "aic": float(2 * n_pars - 2 * ll),
        "theta": theta.tolist(),
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "pi": bw["pi"].tolist(),
        "path": path.astype(np.int8),
    }


def fit_model(data: np.ndarray, ls=LS_DEFAULT, emit_range=None, **kw) -> dict:
    """Fit every candidate ``l`` and select by BIC."""
    if emit_range is None:
        lo, hi = np.percentile(data, [0.5, 99.5])
        emit_range = (float(lo), float(hi))
    fits = {l: fit_trace(data, l, emit_range, **kw) for l in ls}
    best = min(fits, key=lambda l: fits[l]["bic"])
    return {"best_l": int(best), "fits": fits}


def _worker(args):
    i, trace, ls, kwargs = args
    out = fit_model(trace, ls, **kwargs)
    return i, out["best_l"], out["fits"]


def run_split(split: str, ls=LS_DEFAULT, limit: int = 0, jobs: int = 10,
              max_it: int = 100) -> dict:
    data = core.load_traces(split)
    X, y, N = data["X"], data["y"], data["N"]
    n = len(X) if not limit else min(limit, len(X))
    t0 = time.time()
    best = np.zeros(n, dtype=np.int64)
    paths = np.full((n, X.shape[1]), -1, dtype=np.int8)
    lls = {int(l): np.full(n, np.nan) for l in ls}
    bics = {int(l): np.full(n, np.nan) for l in ls}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, bl, fits in ex.map(_worker, [(i, X[i], tuple(ls), {"max_it": max_it}) for i in range(n)],
                                  chunksize=4):
            best[i] = bl
            paths[i] = fits[bl]["path"]
            for l in ls:
                lls[int(l)][i] = fits[l]["loglik"]
                bics[int(l)][i] = fits[l]["bic"]
    runtime = time.time() - t0

    open_true = y[:n]
    open_hat = paths
    n_acc = float((best == N[:n]).mean())
    open_acc = float((open_hat == open_true).mean())
    open_mae = float(np.abs(open_hat - open_true).mean())
    per_n = {}
    for k in range(1, 6):
        m = N[:n] == k
        if m.any():
            per_n[int(k)] = {
                "traces": int(m.sum()),
                "n_acc": float((best[m] == N[:n][m]).mean()),
                "open_acc": float((open_hat[m] == open_true[m]).mean()),
                "open_mae": float(np.abs(open_hat[m] - open_true[m]).mean()),
            }
    return {
        "method": "vnd_port",
        "split": split,
        "traces": int(n),
        "runtime_s": round(runtime, 1),
        "ls": [int(l) for l in ls],
        "metrics": {
            "n_acc": n_acc,
            "open_acc": open_acc,
            "open_mae": open_mae,
            "per_n": per_n,
            "bic_selected_l_hist": {str(k): int((best == k).sum()) for k in range(1, 6)},
        },
        "paths": paths,
        "N_hat": best.tolist(),
        "N_true": N[:n].tolist(),
        "loglik": {str(l): lls[l].tolist() for l in ls},
        "bic": {str(l): bics[l].tolist() for l in ls},
    }


def simulate(theta: np.ndarray, n_channels: int, n_samples: int,
             rng: np.random.Generator, delta: float = 1.0,
             sigma: float = 0.1, level0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Sample a VND path and Gaussian observations (levels 0, 1, ..., l)."""
    T = vnd_transition(theta, n_channels)
    s = np.empty(n_samples, dtype=np.int64)
    s[0] = rng.integers(n_channels + 1)
    for t in range(1, n_samples):
        s[t] = rng.choice(n_channels + 1, p=T[s[t - 1]])
    y = level0 + s + sigma * rng.standard_normal(n_samples)
    return s, y


def verify() -> None:
    rng = np.random.default_rng(0)
    for l in range(1, 6):
        for _ in range(20):
            theta = rng.uniform(0.05, 0.99, size=2 * l)
            T = vnd_transition(theta, l)
            assert np.allclose(T.sum(axis=1), 1.0, atol=1e-12)
    T_id = vnd_transition(np.ones(4), 2)
    assert np.allclose(T_id, np.eye(3), atol=1e-12)
    T_ind = vnd_transition(np.array([0.9, 0.8, 0.9, 0.8]), 2)
    assert np.allclose(T_ind, core.binomial_transition(2, 0.9, 0.8), atol=1e-12)
    print("transition matrix properties ok")

    theta = np.array([0.99, 0.98, 0.98, 0.99])
    s, y = simulate(theta, 2, 20000, rng)
    res = fit_trace(y, 2, (float(y.min()), float(y.max())))
    print("recovery l=2:", np.round(res["theta"], 3), "true", theta)
    assert np.max(np.abs(np.asarray(res["theta"]) - theta)) < 0.03
    sel = fit_model(y, ls=(1, 2, 3), emit_range=(float(y.min()), float(y.max())))
    print("BIC selected l:", sel["best_l"])
    assert sel["best_l"] == 2

    s3, y3 = simulate(np.array([0.95, 0.9, 0.85, 0.9, 0.95, 0.98]), 3, 5000, rng)
    path = core.viterbi(y3[:200], np.full(4, 0.25),
                        vnd_transition([0.95, 0.9, 0.85, 0.9, 0.95, 0.98], 3),
                        np.arange(4.0), np.full(4, 0.01))
    path_ref = core.viterbi_bruteforce(y3[:200], np.full(4, 0.25),
                                       vnd_transition([0.95, 0.9, 0.85, 0.9, 0.95, 0.98], 3),
                                       np.arange(4.0), np.full(4, 0.01))
    assert np.array_equal(path, path_ref)
    print("viterbi matches brute force")
    print("vnd_port verify: all checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--split", default="test_x1")
    parser.add_argument("--ls", default=",".join(str(l) for l in LS_DEFAULT))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--max-it", type=int, default=100)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    if args.verify:
        verify()
        return

    ls = tuple(int(v) for v in args.ls.split(","))
    report = run_split(args.split, ls=ls, limit=args.limit, jobs=args.jobs,
                       max_it=args.max_it)
    paths = report.pop("paths")
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"vnd_{args.split}"
    (RESULTS / f"{tag}.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(RESULTS / f"{tag}_paths.npz", paths=paths,
                        N_hat=report["N_hat"], N_true=report["N_true"])
    m = report["metrics"]
    print(f"{args.split}: {report['traces']} traces in {report['runtime_s']}s | "
          f"N={m['n_acc']:.3f} open={m['open_acc']:.4f} MAE={m['open_mae']:.4f}")
    print("BIC selected l histogram:", m["bic_selected_l_hist"])
    print("saved", RESULTS / f"{tag}.json")


if __name__ == "__main__":
    main()

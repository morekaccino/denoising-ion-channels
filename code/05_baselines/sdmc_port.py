"""Python port of the SD-HMM (Requadt & Li 2026) with AIC/BIC model selection.

Reference implementation: https://gitlab.gwdg.de/requadt/sdmc (GPL-3), files
``src/sd_model.cpp`` (rate matrix, matrix exponential), ``R/HMM_estimation.R``
(``SD_HMM_estimate``, ``bic_aic_sd_hmm``, ``normal_params``,
``normal_probabilities``), ``src/Baum_Welch_step.cpp``,
``src/Likelihood_Forward.cpp``, ``src/Viterbi_simple.cpp``.

The model is a continuous-time sum-dependent Markov chain on the open-channel
count ``S in {0..L}`` with rates ``(lambda_0..lambda_{L-1}, mu_1..mu_L)``,
sampled at interval ``delta`` (transition matrix ``exp(delta * R)``), Gaussian
emissions with equidistant means and per-level standard deviations.

Usage:
  python code/05_baselines/sdmc_port.py --verify
  python code/05_baselines/sdmc_port.py --split test_x1 --limit 16
  python code/05_baselines/sdmc_port.py --split test_x1 --jobs 10
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hmm_core as core  # noqa: E402

RESULTS = core.ROOT / "code" / "05_baselines" / "results"
LS_DEFAULT = (1, 2, 3, 4, 5)


def rate_matrix(theta: np.ndarray, n_channels: int) -> np.ndarray:
    """Port of ``rate_mat``: birth-death rate matrix of the sum process."""
    theta = np.asarray(theta, dtype=np.float64)
    L = n_channels
    R = np.zeros((L + 1, L + 1))
    for i in range(L + 1):
        if i < L:
            R[i, i + 1] = (L - i) * theta[i]
        if i > 0:
            R[i, i - 1] = i * theta[L + i - 1]
    R[np.diag_indices(L + 1)] = -R.sum(axis=1)
    return R


def theta_m_step(xi: np.ndarray, n_channels: int, delta: float,
                 theta0: np.ndarray, max_rate: float = 1e5,
                 trunc_value: float = 1e-6) -> np.ndarray:
    """Port of ``matrix_likelihood``/``log_cost``: Nelder-Mead on log rates."""
    target = np.asarray(xi).ravel()

    def cost(log_theta):
        theta = np.exp(log_theta)
        P = np.clip(expm(delta * rate_matrix(theta, n_channels)), 1e-10, None)
        return float(-np.sum(target * np.log(P.ravel())))

    res = minimize(cost, np.log(np.asarray(theta0, dtype=np.float64)),
                   method="Nelder-Mead",
                   options={"maxiter": 500, "maxfev": 1000, "xatol": 1e-6, "fatol": 1e-8})
    theta = np.exp(res.x)
    theta[theta > max_rate] = trunc_value
    return theta


def fit_trace(data: np.ndarray, n_channels: int, delta: float,
              emit_range: tuple[float, float], max_it: int = 100,
              threshold: float = 1e-6, max_rate: float = 1e5) -> dict:
    """Port of ``SD_HMM_estimate`` for one fixed channel count."""
    n_levels = n_channels + 1
    step = (emit_range[1] - emit_range[0]) / (n_levels - 1)
    if not np.isfinite(step) or step <= 0:
        step = 1.0
    emit = np.concatenate([[emit_range[0], step], np.full(n_levels, 10.0 / step)])
    theta = np.full(2 * n_channels, 10.0)
    pi_current = np.full(n_levels, 1.0 / n_levels)
    P = expm(delta * rate_matrix(theta, n_channels))
    bw = None
    for _ in range(max_it):
        probs = core.gaussian_level_probabilities(data, emit, n_levels)
        bw = core.bw_step(data, probs, P, init=pi_current)
        pi_current = bw["pi"]
        emit_new = core.normal_params(bw["gamma"], bw["y_gamma"], bw["y_gamma2"],
                                      n_levels, variant="sdmc")
        theta_new = theta_m_step(bw["xi"], n_channels, delta, theta, max_rate=max_rate)
        P_new = expm(delta * rate_matrix(theta_new, n_channels))
        err = max(np.abs(P_new - P).max(), np.abs(emit_new - emit).max())
        emit, theta, P = emit_new, theta_new, P_new
        if err < threshold:
            break

    mu = emit[0] + np.arange(n_levels) * emit[1]
    sigma = 1.0 / np.abs(emit[2:])
    ll = core.forward_loglik(data, pi_current, core.gaussian_densities(data, mu, sigma), P)
    path = core.viterbi(data, pi_current, P, mu, sigma**2)
    n_pars = 3 * n_channels + 3
    n_obs = len(data)
    return {
        "L": n_channels,
        "loglik": float(ll),
        "k": n_pars,
        "bic": float(n_pars * np.log(n_obs) - 2 * ll),
        "aic": float(2 * n_pars - 2 * ll),
        "theta": theta.tolist(),
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "pi": pi_current.tolist(),
        "path": path.astype(np.int8),
    }


def fit_model(data: np.ndarray, ls=LS_DEFAULT, delta: float = 0.01,
              emit_range=None, **kw) -> dict:
    if emit_range is None:
        lo, hi = np.percentile(data, [0.5, 99.5])
        emit_range = (float(lo), float(hi))
    fits = {l: fit_trace(data, l, delta, emit_range, **kw) for l in ls}
    best = min(fits, key=lambda l: fits[l]["bic"])
    return {"best_l": int(best), "fits": fits}


def _worker(args):
    i, trace, ls, kwargs = args
    out = fit_model(trace, ls, **kwargs)
    return i, out["best_l"], out["fits"]


def run_split(split: str, ls=LS_DEFAULT, delta: float = 0.01, limit: int = 0,
              jobs: int = 10, max_it: int = 100) -> dict:
    data = core.load_traces(split)
    X, y, N = data["X"], data["y"], data["N"]
    n = len(X) if not limit else min(limit, len(X))
    t0 = time.time()
    best = np.zeros(n, dtype=np.int64)
    paths = np.full((n, X.shape[1]), -1, dtype=np.int8)
    lls = {int(l): np.full(n, np.nan) for l in ls}
    bics = {int(l): np.full(n, np.nan) for l in ls}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, bl, fits in ex.map(_worker, [(i, X[i], tuple(ls), {"delta": delta, "max_it": max_it})
                                            for i in range(n)], chunksize=4):
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
        "method": "sdmc_port",
        "split": split,
        "traces": int(n),
        "runtime_s": round(runtime, 1),
        "delta": delta,
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


def simulate(theta: np.ndarray, n_channels: int, n_samples: int, delta: float,
             rng: np.random.Generator, sigma: float = 0.1,
             level0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Sample an SDMC sum path and Gaussian observations."""
    P = expm(delta * rate_matrix(theta, n_channels))
    s = np.empty(n_samples, dtype=np.int64)
    s[0] = rng.integers(n_channels + 1)
    for t in range(1, n_samples):
        s[t] = rng.choice(n_channels + 1, p=P[s[t - 1]])
    y = level0 + s + sigma * rng.standard_normal(n_samples)
    return s, y


def verify() -> None:
    rng = np.random.default_rng(1)
    for L in range(1, 6):
        theta = rng.uniform(1.0, 20.0, size=2 * L)
        R = rate_matrix(theta, L)
        assert np.allclose(R.sum(axis=1), 0.0, atol=1e-12)
        for i in range(L):
            assert np.isclose(R[i, i + 1], (L - i) * theta[i])
        for i in range(1, L + 1):
            assert np.isclose(R[i, i - 1], i * theta[L + i - 1])
    P = expm(0.05 * rate_matrix(np.array([3.0, 4.0, 4.0, 3.0]), 2))
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)
    print("rate matrix + embedding properties ok")

    theta = np.array([3.0, 4.0, 4.0, 3.0])
    s, y = simulate(theta, 2, 20000, 0.05, rng)
    res = fit_trace(y, 2, 0.05, (float(y.min()), float(y.max())))
    est = np.asarray(res["theta"])
    print("recovery L=2:", np.round(est, 2), "true", theta)
    assert np.all(np.abs(est - theta) / theta < 0.25)

    theta3 = np.array([1.0, 5.0, 9.0, 9.0, 5.0, 1.0])
    _, y3 = simulate(theta3, 3, 20000, 0.05, rng)
    res3 = fit_trace(y3, 3, 0.05, (float(y3.min()), float(y3.max())))
    est3 = np.asarray(res3["theta"])
    opening = est3[:3]
    assert opening[0] < opening[1] < opening[2]
    print("positive cooperativity recovered:", np.round(opening, 2))

    theta_neg = np.array([9.0, 5.0, 1.0, 1.0, 5.0, 9.0])
    _, yn = simulate(theta_neg, 3, 20000, 0.05, rng)
    resn = fit_trace(yn, 3, 0.05, (float(yn.min()), float(yn.max())))
    opening_n = np.asarray(resn["theta"])[:3]
    assert opening_n[0] > opening_n[1] > opening_n[2]
    print("negative cooperativity recovered:", np.round(opening_n, 2))

    sel = fit_model(y, ls=(1, 2, 3), delta=0.05, emit_range=(float(y.min()), float(y.max())))
    print("BIC selected L:", sel["best_l"])
    assert sel["best_l"] == 2

    P3 = expm(0.05 * rate_matrix(theta3, 3))
    mu = np.arange(4.0)
    path = core.viterbi(y3[:200], np.full(4, 0.25), P3, mu, np.full(4, 0.01))
    path_ref = core.viterbi_bruteforce(y3[:200], np.full(4, 0.25), P3, mu, np.full(4, 0.01))
    assert np.array_equal(path, path_ref)
    print("viterbi matches brute force")
    print("sdmc_port verify: all checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--split", default="test_x1")
    parser.add_argument("--ls", default=",".join(str(l) for l in LS_DEFAULT))
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--max-it", type=int, default=100)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    if args.verify:
        verify()
        return

    ls = tuple(int(v) for v in args.ls.split(","))
    report = run_split(args.split, ls=ls, delta=args.delta, limit=args.limit,
                       jobs=args.jobs, max_it=args.max_it)
    paths = report.pop("paths")
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"sdmc_{args.split}"
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

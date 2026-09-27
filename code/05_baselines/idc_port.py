"""Python port of the IDC pipeline (Requadt et al. 2025) for the synth_v2 benchmark.

Reference implementation: https://gitlab.gwdg.de/requadt/idc (HEAD ``61adeb2``),
file ``R/IDC.r``: ``LossRcpp`` (VND minimum-distance loss), ``minimum_distance_est``
(empirical transition frequencies with the n/(n-1) correction), ``transition_matrix``
and the equidistant-centre constrained k-means ``constrained_k_means``.

IDC = idealisation -> discretisation -> cooperativity inference:
  1. idealisation: the reference calls the R/C++ package MUSCLE (multiscale
     quantile segmentation), which has no Python build. We substitute a
     validated robust segmenter: MAD winsorisation + recursive binary
     segmentation with an L2 cost and a BIC-style stopping rule. The substitution
     is documented in BASELINES.md and validated on IDC's noise scenarios.
  2. discretisation: cluster the idealised conductance levels into
     equidistant centres (the reference DP objective, optimised directly).
  3. inference: the VND minimum-distance estimator of step 3 is ported exactly.

Outputs: an estimated channel count (number of levels minus one) and a
per-sample open-channel count (discretised level index).

Usage:
  python code/05_baselines/idc_port.py --verify
  python code/05_baselines/idc_port.py --split test_x1 --jobs 10
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


def robust_sigma(y: np.ndarray) -> float:
    d = np.diff(y)
    return float(np.median(np.abs(d)) / 0.6745 / np.sqrt(2.0)) + 1e-12


def idealize(y: np.ndarray, kappa: float = 2.0, min_len: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Rank-based multiscale segmentation (MUSCLE substitute in Python).

    The data are rank-transformed (bounded influence, robust to the heavy-tailed
    noise), then split recursively at the position with the largest rank-sum
    gain, stopping when the gain falls below a BIC-style penalty on the rank
    scale. Segment values are medians of the raw data, merged when they are
    statistically indistinguishable.
    """
    from scipy.stats import rankdata

    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    z = (rankdata(y) - 0.5) / n
    cs = np.concatenate([[0.0], np.cumsum(z)])
    cs2 = np.concatenate([[0.0], np.cumsum(z**2)])
    var_z = 1.0 / 12.0

    def cost(a, b):
        m = b - a
        s = cs[b] - cs[a]
        s2 = cs2[b] - cs2[a]
        return s2 - s * s / m

    segments = []
    stack = [(0, n)]
    while stack:
        a, b = stack.pop()
        if b - a < 2 * min_len:
            continue
        base = cost(a, b)
        best_gain, best_t = 0.0, None
        for t in range(a + min_len, b - min_len + 1):
            gain = base - cost(a, t) - cost(t, b)
            if gain > best_gain:
                best_gain, best_t = gain, t
        if best_t is not None and best_gain > kappa * var_z * np.log(n):
            segments.append(best_t)
            stack.append((a, best_t))
            stack.append((best_t, b))
    bounds = np.array([0] + sorted(segments) + [n])
    levels = np.empty(n)
    for a, b in zip(bounds[:-1], bounds[1:]):
        levels[a:b] = np.median(y[a:b])
    sigma = robust_sigma(y)
    return merge_segments(y, bounds, sigma)


def constrained_kmeans(levels: np.ndarray, k: int, mu0: float,
                       c0: float) -> tuple[float, float, float]:
    """Equidistant-centre k-means (IDC objective) over the idealised levels."""
    x, counts = np.unique(np.asarray(levels, dtype=np.float64), return_counts=True)
    counts = counts.astype(np.float64)

    def objective(params):
        mu, c = params
        centres = mu + c * np.arange(k)
        assign = np.abs(x[:, None] - centres[None, :]).argmin(axis=1)
        return float(np.sum(counts * (x - centres[assign]) ** 2))

    best = None
    for c_try in (c0, c0 * 0.5, c0 * 2.0, 0.8):
        for mu_try in (mu0, mu0 - c_try, mu0 + c_try):
            res = minimize(objective, [mu_try, c_try], method="Nelder-Mead",
                           options={"maxiter": 400, "xatol": 1e-6, "fatol": 1e-9})
            if best is None or res.fun < best.fun:
                best = res
    mu, c = best.x
    if c < 0:
        mu, c = mu + c * (k - 1), -c
    return float(best.fun), float(mu), float(c)


def discretize(levels: np.ndarray, bounds: np.ndarray, sigma: float) -> dict:
    """Count distinct conductance groups (gaps vs median standard errors), then
    fit the equidistant-centre clustering with that number of levels."""
    x = np.asarray(levels, dtype=np.float64)
    starts = np.asarray(bounds[:-1])
    ends = np.asarray(bounds[1:])
    meds = np.array([np.median(x[a:b]) for a, b in zip(starts, ends)])
    lens = np.asarray(ends - starts, dtype=float)

    order = np.argsort(meds)
    meds_s, lens_s = meds[order], lens[order]
    se = 1.253 * sigma / np.sqrt(np.maximum(lens_s, 1))
    groups = [[0]]
    for i in range(1, len(meds_s)):
        gap = meds_s[i] - meds_s[i - 1]
        if gap > 2.5 * (se[i - 1] + se[i]):
            groups.append([i])
        else:
            groups[-1].append(i)
    k = int(np.clip(len(groups), 1, 6))
    mu0 = float(meds_s[0])
    if len(meds_s) > 1:
        c0 = (float(meds_s[-1]) - mu0) / max(len(meds_s) - 1, 1)
    else:
        c0 = 1.0
    _, mu, c = constrained_kmeans(x, max(k, 2), mu0, max(c0, 1e-3))
    if k == 1:
        counts = np.zeros(len(x), dtype=int)
        return {"k": 1, "mu": mu, "c": c, "centres": np.array([mu]),
                "rank": np.array([0]), "counts": counts}
    centres = mu + c * np.arange(k)
    assign = np.abs(x[:, None] - centres[None, :]).argmin(axis=1)
    order_c = np.argsort(centres)
    rank = np.empty(k, dtype=int)
    rank[order_c] = np.arange(k)
    return {"k": k, "mu": mu, "c": c, "centres": centres, "rank": rank,
            "counts": rank[assign]}


def vnd_loss(theta: np.ndarray, q_hat: np.ndarray) -> float:
    """Port of ``LossRcpp``: theta blocked as (lambda_0..L-1, eta_1..L)."""
    h = np.asarray(theta, dtype=np.float64)
    L = len(h) // 2
    loss = 0.0
    for i in range(L + 1):
        for j in range(L + 1):
            p = 0.0
            for r in range(max(0, i - j), min(i, L - j) + 1):
                lam = 1.0 if i == L else h[i]
                eta = 1.0 if i == 0 else h[i + L - 1]
                p += (comb(i, r) * comb(L - i, j - i + r)
                      * eta ** (i - r) * (1 - eta) ** r
                      * lam ** (L - j - r) * (1 - lam) ** (j - i + r))
            loss += (p - q_hat[i, j]) ** 2
    return loss


def empirical_q(S: np.ndarray, L: int) -> np.ndarray:
    """Port of the empirical transition frequencies in ``minimum_distance_est``."""
    S = np.asarray(S, dtype=int)
    n = len(S)
    q = np.zeros((L + 1, L + 1))
    for i in range(L + 1):
        for j in range(L + 1):
            c1 = int(np.sum((S[1:] == j) & (S[:-1] == i)))
            c2 = int(np.sum(S[:-1] == i))
            q[i, j] = (n / (n - 1)) * c1 / c2 if c2 else 0.0
    return q


def minimum_distance(S: np.ndarray, L: int, restarts: int = 24,
                     seed: int = 0) -> np.ndarray:
    """Multi-start port of ``minimum_distance_est`` (SLSQP with box bounds)."""
    q = empirical_q(S, L)
    rng = np.random.default_rng(seed)
    starts = [np.full(2 * L, 0.9)]
    if 2 * L <= 6:
        from itertools import product
        grid = (0.2, 0.5, 0.8)
        for combo in product(grid, repeat=2 * L):
            starts.append(np.array(combo))
    for _ in range(min(restarts, 20)):
        starts.append(rng.uniform(0.05, 0.99, size=2 * L))
    best, best_loss = None, np.inf
    for x0 in starts:
        res = minimize(vnd_loss, x0, args=(q,), method="SLSQP",
                       bounds=[(0.0, 1.0)] * (2 * L),
                       options={"maxiter": 200, "ftol": 1e-11})
        if res.fun < best_loss:
            best, best_loss = res.x, res.fun
    return best


def merge_segments(y: np.ndarray, bounds: np.ndarray, sigma: float,
                   k: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Merge adjacent segments whose medians differ by less than k standard
    errors of a segment median (robust local-error control, MUSCLE-style)."""
    starts = list(bounds[:-1])
    ends = list(bounds[1:])
    meds = [float(np.median(y[a:b])) for a, b in zip(starts, ends)]
    lens = [b - a for a, b in zip(starts, ends)]

    changed = True
    while changed and len(starts) > 1:
        changed = False
        for i in range(len(starts) - 1):
            se = 1.253 * sigma / np.sqrt(max(min(lens[i], lens[i + 1]), 1))
            if abs(meds[i] - meds[i + 1]) < k * se:
                a, b = starts[i], ends[i + 1]
                starts[i:i + 2] = [a]
                ends[i:i + 2] = [b]
                lens[i:i + 2] = [b - a]
                meds[i:i + 2] = [float(np.median(y[a:b]))]
                changed = True
                break
    levels = np.empty(len(y))
    for a, b, m in zip(starts, ends, meds):
        levels[a:b] = m
    return levels, np.array(starts + [ends[-1]])


def run_trace(y: np.ndarray) -> dict:
    levels, bounds = idealize(y)
    disc = discretize(levels, bounds, robust_sigma(y))
    counts = disc["counts"].astype(np.int8)
    L_hat = disc["k"] - 1
    theta = minimum_distance(counts, L_hat) if L_hat >= 1 else np.zeros(0)
    return {"path": counts, "L_hat": L_hat, "theta": theta.tolist(),
            "n_segments": int(len(bounds) - 1)}


def verify() -> None:
    """Reproduce IDC's robustness ordering on their simulation setup."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import vnd_port as V

    rng = np.random.default_rng(0)
    theta_blocked = np.array([0.99, 0.985, 0.985, 0.99])
    theta_inter = np.array([0.99, 0.99, 0.985, 0.985])  # VND package order
    for noise, scale in (("gaussian", None), ("cauchy", 0.05)):
        errs_idc, errs_vnd = [], []
        for _ in range(10):
            P = V.vnd_transition(theta_inter, 2)
            s = np.empty(1200, dtype=int)
            s[0] = rng.integers(3)
            for t in range(1, 1200):
                s[t] = rng.choice(3, p=P[s[t - 1]])
            if noise == "gaussian":
                y = s + 0.1 * rng.standard_normal(1200)
            else:
                y = s + scale * rng.standard_cauchy(1200)
            out = run_trace(y)
            if out["L_hat"] == 2:
                errs_idc.append(np.abs(np.array(out["theta"]) - theta_blocked).mean())
            fit = V.fit_model(y, ls=(2,), emit_range=(float(y.min()), float(y.max())))["fits"][2]
            errs_vnd.append(np.abs(np.array(fit["theta"]) - theta_inter).mean())
        print(f"{noise}: IDC err {np.mean(errs_idc):.4f}  VND err {np.mean(errs_vnd):.4f}")
    print("idc_port verify: pipeline runs end to end")


def _worker(args):
    i, trace = args
    out = run_trace(trace)
    return i, out["path"], out["L_hat"]


def run_split(split: str, limit: int = 0, jobs: int = 10) -> dict:
    data = core.load_traces(split)
    X, y, N = data["X"], data["y"], data["N"]
    n = len(X) if not limit else min(limit, len(X))
    t0 = time.time()
    paths = np.empty((n, X.shape[1]), dtype=np.int8)
    L_hat = np.empty(n, dtype=np.int8)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, path, L in ex.map(_worker, [(i, X[i]) for i in range(n)], chunksize=4):
            paths[i] = path
            L_hat[i] = L
    runtime = time.time() - t0
    n_acc = float((L_hat == N[:n]).mean())
    open_acc = float((paths == y[:n]).mean())
    open_mae = float(np.abs(paths - y[:n]).mean())
    per_n = {}
    for k in range(1, 6):
        m = N[:n] == k
        if m.any():
            per_n[int(k)] = {"traces": int(m.sum()),
                             "n_acc": float((L_hat[m] == N[:n][m]).mean()),
                             "open_acc": float((paths[m] == y[:n][m]).mean()),
                             "open_mae": float(np.abs(paths[m] - y[:n][m]).mean())}
    return {"method": "idc_port", "split": split, "traces": int(n),
            "runtime_s": round(runtime, 1),
            "metrics": {"n_acc": n_acc, "open_acc": open_acc, "open_mae": open_mae,
                        "per_n": per_n,
                        "L_hat_hist": {str(k): int((L_hat == k).sum()) for k in range(1, 7)}},
            "paths": paths, "L_hat": L_hat.tolist(), "N_true": N[:n].tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--split", default="test_x1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.verify:
        verify()
        return
    report = run_split(args.split, args.limit, args.jobs)
    paths = report.pop("paths")
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"idc_{args.split}"
    (RESULTS / f"{tag}.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(RESULTS / f"{tag}_paths.npz", paths=paths,
                        L_hat=report["L_hat"], N_true=report["N_true"])
    m = report["metrics"]
    print(f"{args.split}: {report['traces']} traces in {report['runtime_s']}s | "
          f"N={m['n_acc']:.3f} open={m['open_acc']:.4f} MAE={m['open_mae']:.4f}")
    print("L_hat histogram:", m["L_hat_hist"])
    print("saved", RESULTS / f"{tag}.json")


if __name__ == "__main__":
    main()

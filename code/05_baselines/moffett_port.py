"""Vectorized port of the Moffett et al. 2022 CFTR factor-graph EM.

Reference implementation: https://doi.org/10.5281/zenodo.7073043
(`andreweckford/PatchClampFactorGraphEM`, license other-open), files
``factor/SumProductNodes.py``, ``receptors/CFTR.py`` and ``main.py``.

The reference is a sum-product / factor-graph EM for a single channel with
Gaussian observations on the fixed 7-state CFTR model. Their state order and
default rate table are identical to this repo's (C1a, C1b, C2, O1, O2, C3, C4;
statemap 0,0,0,1,1,0,0). This module reuses their EM exactly but vectorized
over time so the 69 single-channel test traces can be processed:

  E-step: scaled forward-backward messages, normalized at variable nodes
  (as in their StateNode), stationary init on the left, ones on the right.
  M-step: Q from normalized pairwise posteriors summed over t=1..T-2 and
  row-normalized (their loop range and normalization are preserved),
  amplitudes as posterior-weighted means per conductance level, sigma2 as
  the posterior-weighted mean squared residual (their equations verbatim).

``--verify`` checks the port against their node implementation on random
short traces.

Usage:
  python code/05_baselines/moffett_port.py --verify
  python code/05_baselines/moffett_port.py --split test_x1 --jobs 10
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hmm_core as core  # noqa: E402

ROOT = core.ROOT
MOFFETT_CODE = ROOT / ".external" / "moffett" / "code"
RESULTS = ROOT / "code" / "05_baselines" / "results"
N_STATES = 7
STATEMAP = np.array([0, 0, 0, 1, 1, 0, 0])


def stationary(P: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    v = np.abs(v)
    return v / v.sum()


def emission_matrix(y: np.ndarray, amplitude: np.ndarray, sigma2: float,
                    statemap: np.ndarray = STATEMAP) -> np.ndarray:
    """IonChannelNodeNoisy.message for every time step, shape (T, 7)."""
    means = amplitude[statemap.astype(int)]
    z = (y[:, None] - means[None, :])
    return np.exp(-0.5 * z**2 / sigma2) / np.sqrt(2 * np.pi * sigma2)


def fit_trace(y: np.ndarray, max_iter: int = 10) -> dict:
    """Port of their main() EM loop for one trace (no ground-truth inputs)."""
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    P = np.array([
        [-9.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [5.0, -12.7, 7.7, 0.0, 0.0, 0.0, 0.0],
        [0.0, 5.8, -10.7, 4.9, 0.0, 0.0, 0.0],
        [0.0, 0.0, 10.0, -17.1, 7.1, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -3.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 7.0, -13.0, 6.0],
        [1.7, 0.0, 0.0, 0.0, 0.0, 12.8, -14.5],
    ])
    dt = 0.01
    from scipy.linalg import expm
    P = expm(dt * P)

    lo, hi = np.percentile(y, [10, 90])
    amplitude = np.array([lo, hi])
    sigma2 = float(np.var(y))

    posterior = None
    for _ in range(max_iter):
        b = emission_matrix(y, amplitude, sigma2)
        pi = stationary(P)

        right_in = np.empty((T, N_STATES))
        right_out = np.empty((T, N_STATES))
        right_in[0] = pi
        right_out[0] = pi * b[0]
        right_out[0] /= right_out[0].sum()
        for t in range(1, T):
            right_in[t] = right_out[t - 1] @ P
            right_out[t] = right_in[t] * b[t]
            right_out[t] /= right_out[t].sum()

        left_in = np.empty((T, N_STATES))
        left_out = np.empty((T, N_STATES))
        left_in[T - 1] = np.ones(N_STATES)
        left_out[T - 1] = left_in[T - 1] * b[T - 1]
        left_out[T - 1] /= left_out[T - 1].sum()
        for t in range(T - 2, -1, -1):
            left_in[t] = P @ left_out[t + 1]
            left_out[t] = left_in[t] * b[t]
            left_out[t] /= left_out[t].sum()

        posterior = right_in * b * left_in
        posterior = posterior / posterior.sum(axis=1, keepdims=True)
        no_channel = right_in * left_in
        no_channel = no_channel / no_channel.sum(axis=1, keepdims=True)

        q = np.zeros((N_STATES, N_STATES))
        for t in range(1, T - 1):
            pair = right_out[t][:, None] * P * left_out[t + 1][None, :]
            q += pair / pair.sum()
        q = q / q.sum(axis=1, keepdims=True)
        P = q

        p_closed = no_channel[:, STATEMAP == 0].sum(axis=1)
        p_open = no_channel[:, STATEMAP == 1].sum(axis=1)
        amplitude = np.array([
            np.sum(y * p_closed) / max(p_closed.sum(), 1e-12),
            np.sum(y * p_open) / max(p_open.sum(), 1e-12),
        ])
        sigma2 = float(np.mean((y - amplitude[0]) ** 2 * p_closed
                               + (y - amplitude[1]) ** 2 * p_open))

    path = posterior.argmax(axis=1)
    return {"path": path.astype(np.int8), "P": P, "amplitude": amplitude,
            "sigma2": sigma2, "posterior": posterior}


def verify() -> None:
    """Compare against the original node-based implementation."""
    sys.path.insert(0, str(MOFFETT_CODE))
    import factor.SumProductNodes as sp  # noqa: E402
    from factor.myTools import getSteadyStateDist  # noqa: E402

    rng = np.random.default_rng(0)
    for trial in range(3):
        T = 200
        y = np.concatenate([rng.normal(0.2, 0.15, T // 2),
                            rng.normal(1.1, 0.15, T // 2)])
        rng.shuffle(y)
        out = fit_trace(y, max_iter=5)

        # original implementation, same number of iterations
        statemap = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        P = np.array([
            [-9.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [5.0, -12.7, 7.7, 0.0, 0.0, 0.0, 0.0],
            [0.0, 5.8, -10.7, 4.9, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, -17.1, 7.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -3.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 7.0, -13.0, 6.0],
            [1.7, 0.0, 0.0, 0.0, 0.0, 12.8, -14.5]])
        from scipy.linalg import expm
        P = expm(0.01 * P)
        current = np.array([np.percentile(y, 10), np.percentile(y, 90)])
        sigma2 = float(np.var(y))
        for _ in range(5):
            v, s, c = [], [], []
            b = emission_matrix(y, current, sigma2)
            for t in range(T):
                v.append(sp.StateNode())
                c.append(sp.IonChannelNodeNoisy(y[t], statemap, current, sigma2))
                v[t].setChannelMessage(b[t])
            for t in range(T - 1):
                s.append(sp.MarkovFactorNode(P))
            v[0].setRightInMessage(getSteadyStateDist(P))
            s[0].setRightInMessage(v[0].rightOutMessage())
            for t in range(1, T - 1):
                v[t].setRightInMessage(s[t - 1].rightOutMessage())
                s[t].setRightInMessage(v[t].rightOutMessage())
            v[T - 1].setRightInMessage(s[T - 2].rightOutMessage())
            v[T - 1].setLeftInMessage(np.ones(7))
            for t in range(T - 2, -1, -1):
                s[t].setLeftInMessage(v[t + 1].leftOutMessage())
                v[t].setLeftInMessage(s[t].leftOutMessage())
            q = np.zeros((7, 7))
            for t in range(1, T - 1):
                q += s[t].aPosteriori()
            P = q / q.sum(axis=1, keepdims=True)
            post_nc = np.array([v[t].aPosterioriNoChannel() for t in range(T)])
            closed_mask = np.array(statemap) == 0
            p_closed = post_nc[:, closed_mask].sum(axis=1)
            p_open = post_nc[:, ~closed_mask].sum(axis=1)
            current = np.array([np.sum(y * p_closed) / p_closed.sum(),
                                np.sum(y * p_open) / p_open.sum()])
            sigma2 = float(np.mean((y - current[0]) ** 2 * p_closed
                                   + (y - current[1]) ** 2 * p_open))
        ref = np.array([np.argmax(v[t].aPosteriori()) for t in range(T)])
        assert np.array_equal(out["path"], ref), "path mismatch"
        assert np.allclose(out["P"], P, atol=1e-8), "P mismatch"
        assert np.allclose(out["amplitude"], current, atol=1e-8), "amplitude mismatch"
        assert abs(out["sigma2"] - sigma2) < 1e-8, "sigma2 mismatch"
    print("moffett_port matches the reference node implementation")
    print("moffett_port verify: all checks passed")


def _worker(args):
    i, trace = args
    out = fit_trace(trace)
    return i, out["path"], out["P"], out["amplitude"], out["sigma2"]


def run_split(split: str, limit: int = 0, jobs: int = 10, max_iter: int = 10) -> dict:
    data = core.load_traces(split)
    X, y, r, N = data["X"], data["y"], data["r"], data["N"]
    keep = np.flatnonzero(N == 1)
    if limit:
        keep = keep[:limit]
    n = len(keep)
    states_true = r[keep, 0, :]
    t0 = time.time()
    paths = np.empty((n, X.shape[1]), dtype=np.int8)
    Ps = np.empty((n, N_STATES, N_STATES))
    amps = np.empty((n, 2))
    s2 = np.empty(n)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, path, P, amp, sig in ex.map(_worker, [(j, X[k]) for j, k in enumerate(keep)],
                                            chunksize=2):
            paths[i] = path
            Ps[i] = P
            amps[i] = amp
            s2[i] = sig
    runtime = time.time() - t0
    state_acc = float((paths == states_true).mean())
    open_true = (STATEMAP[states_true] == 1)
    open_hat = (STATEMAP[paths.astype(int)] == 1)
    open_acc = float((open_hat == open_true).mean())
    open_mae = float(np.abs(open_hat.astype(np.float64)
                            - open_true.astype(np.float64)).mean())
    per_state_acc = {}
    for s in range(N_STATES):
        true_is = states_true == s
        if true_is.any():
            per_state_acc[int(s)] = float((paths[true_is] == s).mean())
    return {
        "method": "moffett_port",
        "split": split,
        "subset": "N=1 traces only (single-channel method)",
        "traces": int(n),
        "runtime_s": round(runtime, 1),
        "metrics": {
            "state_acc": state_acc,
            "open_acc": open_acc,
            "open_mae": open_mae,
            "per_state_acc": per_state_acc,
            "amplitude_mean": amps.mean(axis=0).tolist(),
            "sigma2_mean": float(s2.mean()),
        },
        "paths": paths,
        "states_true": states_true.astype(np.int8),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--split", default="test_x1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.verify:
        verify()
        return
    report = run_split(args.split, args.limit, args.jobs, args.max_iter)
    paths = report.pop("paths")
    states_true = report.pop("states_true")
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"moffett_{args.split}"
    (RESULTS / f"{tag}.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(RESULTS / f"{tag}_paths.npz", paths=paths, states_true=states_true)
    m = report["metrics"]
    print(f"{args.split}: {report['traces']} N=1 traces in {report['runtime_s']}s | "
          f"state={m['state_acc']:.4f} open={m['open_acc']:.4f}")
    print("saved", RESULTS / f"{tag}.json")


if __name__ == "__main__":
    main()

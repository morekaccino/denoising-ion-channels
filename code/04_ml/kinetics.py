"""Exact inference for the multi-channel CFTR generative model.

Because the patch clamp sums N iid copies of the same 7-state kinetic chain
(source/ion_channel.py), the vector of occupancy counts is itself an exact
Markov chain with C(N+6, 6) states, and the open count is a deterministic
function of that state. This module provides the exact posterior
p(open count | signal), the marginal likelihood used to select N, and an
EM variant that re-estimates the aggregate transition matrix from unlabeled
data (e.g. real ABF traces).

This is the Bayes-optimal decoder for the simulator and the inference core of
the kinetics-informed neural HMM.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import scipy.stats as st

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import benchmark as B  # noqa: E402


def count_states(N: int) -> np.ndarray:
    states: list[tuple[int, ...]] = []

    def rec(prefix: list[int], remaining: int) -> None:
        if len(prefix) == 6:
            states.append(tuple(prefix + [remaining]))
            return
        for v in range(remaining + 1):
            rec(prefix + [v], remaining - v)

    rec([], N)
    return np.asarray(states, dtype=np.int16)


def count_transition_matrix(N: int, P0: np.ndarray) -> np.ndarray:
    """Exact one-step transition matrix of the occupancy-count chain.

    Each of the N channels independently applies P0 once. Channels sharing a
    source kinetic state have identical kernels, so the count transition is the
    convolution of n_i copies of the categorical kernel P0[i, :].
    """
    states = count_states(N)
    index = {tuple(s): i for i, s in enumerate(states)}
    S = len(states)
    T = np.zeros((S, S))
    zero = np.zeros(7, dtype=np.int16)
    for si, n in enumerate(states):
        dist: dict[tuple[int, ...], float] = {tuple(zero): 1.0}
        for i in range(7):
            for _ in range(int(n[i])):
                new: dict[tuple[int, ...], float] = defaultdict(float)
                row = P0[i]
                for cnt, p in dist.items():
                    for j in np.flatnonzero(row):
                        m = list(cnt)
                        m[j] += 1
                        new[tuple(m)] += p * row[j]
                dist = new
        for cnt, p in dist.items():
            T[si, index[cnt]] = p
    return T


def count_stationary(N: int, P0: np.ndarray) -> np.ndarray:
    pi_single = B.stationary(P0)
    states = count_states(N)
    log_norm = st.multinomial.logpmf(states, N, pi_single)
    return np.exp(log_norm)


def default_grid(N: int, spacing: float = 0.002, pad: float = 2.5, noise_scale: float = 1.0) -> np.ndarray:
    return np.arange(-pad * noise_scale, N * B.OPEN_GH["loc"] + pad * noise_scale, spacing)


def _scaled(params: dict, noise_scale: float) -> dict:
    out = dict(params)
    out["scale"] = params["scale"] * noise_scale
    return out


def emission_log_densities(
    N: int,
    grid: np.ndarray,
    open_params: dict | None = None,
    close_params: dict | None = None,
    noise_scale: float = 1.0,
) -> np.ndarray:
    """Exact log p(x | k open channels) on a common grid.

    The observation is the sum of k open-channel and N-k closed-channel
    generalized-hyperbolic noises, so the density is the k-fold convolution of
    the open pdf with the (N-k)-fold convolution of the closed pdf. Both
    components are heavy-tailed, so both must be convolved (the open level is
    not a point mass).
    """
    open_params = _scaled(open_params or B.OPEN_GH, noise_scale)
    close_params = _scaled(close_params or B.CLOSE_GH, noise_scale)
    h = grid[1] - grid[0]
    x0 = grid[0]
    G = len(grid)

    def pdf(params: dict) -> np.ndarray:
        d = st.genhyperbolic.pdf(grid, params["p"], params["a"], params["b"], loc=params["loc"], scale=params["scale"])
        return np.clip(d, 0.0, None)

    open_pdf = pdf(open_params)
    close_pdf = pdf(close_params)
    offset = int(round((N - 1) * (-x0) / h))
    need = N * (G - 1) + 1 + offset
    Lpad = 1 << int(np.ceil(np.log2(max(need, 2))))
    Fo = np.fft.rfft(open_pdf, Lpad)
    Fc = np.fft.rfft(close_pdf, Lpad)

    logE = np.full((N + 1, G), -np.inf)
    for k in range(N + 1):
        m = N - k
        spec = np.ones(Lpad // 2 + 1, dtype=complex)
        if k:
            spec *= Fo**k
        if m:
            spec *= Fc**m
        conv = np.fft.irfft(spec, Lpad)[:need] * h ** (N - 1)
        dens = np.asarray(conv[offset : offset + G], dtype=float)
        logE[k] = np.log(np.clip(dens, 1e-300, None))
    return logE


def log_emission_matrix(X: np.ndarray, grid: np.ndarray, logE: np.ndarray, states: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(grid, X, side="right"), 1, len(grid) - 1)
    frac = (X - grid[idx - 1]) / (grid[idx] - grid[idx - 1])
    lo = logE[:, idx - 1]
    hi = logE[:, idx]
    log_dens = lo + frac[None, :] * (hi - lo)
    open_count = states[:, B.STATEMAP == 1].sum(axis=1).astype(int)
    return log_dens[open_count].T


def forward_backward(logB: np.ndarray, P: np.ndarray, pi: np.ndarray):
    T, S = logB.shape
    shift = logB.max(axis=1)
    b = np.exp(logB - shift[:, None])
    alpha = np.empty((T, S))
    scales = np.empty(T)
    a = pi * b[0]
    s = a.sum()
    scales[0] = s
    alpha[0] = a / s
    for t in range(1, T):
        a = (alpha[t - 1] @ P) * b[t]
        s = a.sum()
        if s <= 0:
            s = 1e-300
        scales[t] = s
        alpha[t] = a / s
    beta = np.empty((T, S))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (P @ (b[t + 1] * beta[t + 1])) / scales[t + 1]
    gamma = alpha * beta
    gamma /= np.clip(gamma.sum(axis=1, keepdims=True), 1e-300, None)
    log_evidence = float(np.sum(np.log(scales) + shift))
    return alpha, beta, scales, gamma, log_evidence


def decode_open_count(
    X: np.ndarray,
    N: int,
    P0: np.ndarray | None = None,
    grid: np.ndarray | None = None,
    logE: np.ndarray | None = None,
    noise_scale: float = 1.0,
    return_evidence: bool = False,
):
    P0 = B.p0_matrix() if P0 is None else P0
    P = count_transition_matrix(N, P0)
    pi = count_stationary(N, P0)
    states = count_states(N)
    if grid is None or logE is None:
        grid = default_grid(N, noise_scale=noise_scale)
        logE = emission_log_densities(N, grid, noise_scale=noise_scale)
    logB = log_emission_matrix(np.asarray(X, dtype=float), grid, logE, states)
    _, _, _, gamma, log_ev = forward_backward(logB, P, pi)
    kstates = states[:, B.STATEMAP == 1].sum(axis=1).astype(int)
    post = np.zeros((len(X), N + 1))
    np.add.at(post.T, kstates, gamma.T)
    pred = post.argmax(axis=1)
    if return_evidence:
        return pred, post, log_ev
    return pred, post


def model_evidence(
    X_values, N_values=range(1, 6), P0: np.ndarray | None = None, noise_scale: float = 1.0
) -> dict[int, float]:
    P0 = B.p0_matrix() if P0 is None else P0
    out = {}
    for N in N_values:
        _, _, log_ev = decode_open_count(X_values, N, P0=P0, noise_scale=noise_scale, return_evidence=True)
        out[int(N)] = log_ev
    return out


def signal_likelihood(
    X: np.ndarray, N_values=range(1, 6), grid: np.ndarray | None = None, noise_scale: float = 1.0
) -> dict[int, float]:
    P0 = B.p0_matrix()
    pi_single = B.stationary(P0)
    p_open = float(pi_single[B.STATEMAP == 1].sum())
    X = np.asarray(X, dtype=float)
    out = {}
    for N in N_values:
        grid_ = default_grid(N, noise_scale=noise_scale) if grid is None else grid
        logE = emission_log_densities(N, grid_, noise_scale=noise_scale)
        idx = np.clip(np.searchsorted(grid_, X, side="right"), 1, len(grid_) - 1)
        frac = (X - grid_[idx - 1]) / (grid_[idx] - grid_[idx - 1])
        dens = np.exp(logE[:, idx - 1]) * (1 - frac)[None, :] + np.exp(logE[:, idx]) * frac[None, :]
        weights = st.binom.pmf(np.arange(N + 1), N, p_open)
        mix = weights @ np.clip(dens, 1e-300, None)
        out[int(N)] = float(np.log(np.clip(mix, 1e-300, None)).sum())
    return out


def _stationary_power(P: np.ndarray, tol: float = 1e-12, max_iter: int = 10000) -> np.ndarray:
    pi = np.full(P.shape[0], 1.0 / P.shape[0])
    for _ in range(max_iter):
        nxt = pi @ P
        if np.abs(nxt - pi).max() < tol:
            return nxt
        pi = nxt
    return pi


def fit_transition_em(
    X: np.ndarray,
    N: int,
    P_init: np.ndarray,
    grid: np.ndarray,
    logE: np.ndarray,
    n_iter: int = 20,
    tol: float = 1e-5,
):
    states = count_states(N)
    logB = log_emission_matrix(np.asarray(X, dtype=float), grid, logE, states)
    P = P_init.copy()
    prev = -np.inf
    log_ev = None
    for _ in range(n_iter):
        pi = _stationary_power(P)
        alpha, beta, scales, gamma, log_ev = forward_backward(logB, P, pi)
        xi_sum = np.zeros_like(P)
        for t in range(len(X) - 1):
            b_next = np.exp(logB[t + 1] - logB[t + 1].max())
            xi = alpha[t][:, None] * P * (b_next * beta[t + 1])[None, :]
            s = xi.sum()
            if s > 0:
                xi_sum += xi / s
        P_new = xi_sum / np.clip(xi_sum.sum(axis=1, keepdims=True), 1e-300, None)
        prev_delta = log_ev - prev if np.isfinite(prev) else np.inf
        P = P_new
        if prev_delta < tol:
            break
        prev = log_ev
    return P, log_ev


def open_count_posterior_map(post: np.ndarray) -> np.ndarray:
    return post.argmax(axis=1)


def _demo(n_traces: int = 5, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    data = B.generate_split(seed=int(rng.integers(1 << 30)), n_traces=n_traces, channel_choices=(1, 2, 3))
    grid = np.arange(-1.0, 8.0, 0.002)
    for i in range(n_traces):
        X, y, N = data["X_s1"][i], data["y"][i], int(data["N"][i])
        pred, post = decode_open_count(X, N, grid=grid, logE=emission_log_densities(N, grid))
        ev = model_evidence(X, range(1, 6)) if i < 2 else {}
        n_hat = max(ev, key=ev.get) if ev else None
        print(
            f"trace {i}: N={N} acc={np.mean(pred == y):.4f} mean_conf={post.max(axis=1).mean():.3f}"
            + (f" N_hat={n_hat}" if n_hat else "")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        _demo()


if __name__ == "__main__":
    main()

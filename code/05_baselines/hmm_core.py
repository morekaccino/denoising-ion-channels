"""Shared Baum-Welch / Viterbi core for the VND and SDMC Python ports.

Faithful ports of the authors' reference implementations:

  VND  (Vanegas et al. 2024, GPL-2)  https://github.com/ljvanegas/VND
       R/estimate_HMM.R, src/Baum_Welch_step.cpp, src/Likelihood_Forward.cpp,
       src/Viterbi_simple.cpp
  SDMC (Requadt & Li 2026, GPL-3)    https://gitlab.gwdg.de/requadt/sdmc
       R/HMM_estimation.R, src/Baum_Welch_step.cpp, src/Likelihood_Forward.cpp,
       src/Viterbi_simple.cpp, src/sd_model.cpp

The structure intentionally mirrors the reference code (including quirks such
as the normalised emissions used by both packages) so that fitted models match
the reference behavior. Deviations are recorded in ``BASELINES.md``.
"""

from __future__ import annotations

import pathlib

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.special import comb

ROOT = pathlib.Path(__file__).resolve().parents[2]
SYNTH = ROOT / "data" / "derived" / "synth_v2"


def load_traces(split: str) -> dict[str, np.ndarray]:
    """Load a synth_v2 split; ``test_x2``-style names pick the noise scale."""
    if "_x" in split:
        base, scale = split.split("_x")[0], split.split("_x")[1]
    else:
        base, scale = split, "1"
    with np.load(SYNTH / f"{base}.npz") as data:
        key = f"X_s{scale}"
        X = data[key] if key in data.files else data["X"]
        return {
            "X": X.astype(np.float64),
            "y": data["y"].astype(np.int64),
            "N": data["N"].astype(np.int64),
            "r": data["r"].astype(np.int64),
            "R": data["R"].astype(np.float64),
            "group": data["group"].astype(np.int64),
        }


def gaussian_level_probabilities(data: np.ndarray, emit: np.ndarray, n_levels: int) -> np.ndarray:
    """Port of ``normal_probabilities`` (both packages): per-datapoint normalised
    Gaussian level densities, shape ``(n_levels, n_data)``."""
    mu = emit[0] + np.arange(n_levels) * emit[1]
    sigma = 1.0 / np.abs(emit[2:])
    expon = -0.5 * (np.asarray(data)[:, None] - mu[None, :]) ** 2 / (sigma[None, :] ** 2)
    expon = expon - expon.max(axis=1, keepdims=True)
    probs = np.exp(expon) / sigma[None, :]
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs.T


def gaussian_densities(data: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Port of ``emission_densities``: plain Gaussian densities, shape
    ``(n_levels, n_data)``."""
    z = (np.asarray(data)[None, :] - mu[:, None]) / sigma[:, None]
    return np.exp(-0.5 * z**2) / (sigma[:, None] * np.sqrt(2 * np.pi))


def bw_step(data: np.ndarray, probabilities: np.ndarray, P_trans: np.ndarray,
            init: np.ndarray | None = None) -> dict:
    """Port of ``BW_step`` (VND and SDMC variants).

    ``probabilities`` is ``(K, T)``; returns the same quantities as the Rcpp
    code: normalised ``P_trans``, raw ``xi`` weights, per-level ``gamma`` sums,
    per-level ``y_gamma``/``y_gamma2`` and the free initial distribution.
    """
    data = np.asarray(data, dtype=np.float64)
    n_data = len(data)
    n_levels = probabilities.shape[0]
    if init is None:
        init_used = np.full(n_levels, 1.0 / n_levels)
    else:
        init_used = np.asarray(init, dtype=np.float64)
        init_used = init_used / init_used.sum()
    P = np.asarray(P_trans, dtype=np.float64)

    alpha = np.zeros((n_levels, n_data))
    beta = np.zeros((n_levels, n_data))
    prob = probabilities[:, 0] * init_used
    alpha[:, 0] = prob / prob.sum()
    beta[:, -1] = 1.0
    for i in range(n_data - 1):
        j = n_data - 1 - i
        prob = P @ (beta[:, j] * probabilities[:, j])
        beta[:, j - 1] = prob / prob.sum()
        prob = (P.T @ alpha[:, i]) * probabilities[:, i + 1]
        alpha[:, i + 1] = prob / prob.sum()

    gamma = alpha * beta
    gamma = gamma / gamma.sum()
    y_gamma = (gamma * data[None, :]).sum(axis=1)
    y_gamma2 = (gamma * data[None, :] ** 2).sum(axis=1)
    est_pi = gamma[:, 0] / gamma[:, 0].sum()

    est_P = np.zeros((n_levels, n_levels))
    for i in range(n_data - 1):
        eps = (alpha[:, i][:, None] * P) * (probabilities[:, i + 1] * beta[:, i + 1])[None, :]
        est_P = est_P + eps / eps.sum()
    xi = est_P.copy()
    est_P = est_P / est_P.sum(axis=1, keepdims=True)

    return {
        "P_trans": est_P,
        "xi": xi,
        "gamma": gamma.sum(axis=1),
        "y_gamma": y_gamma,
        "y_gamma2": y_gamma2,
        "pi": est_pi,
        "init_used": init_used,
    }


def forward_loglik(data: np.ndarray, init: np.ndarray, probabilities: np.ndarray,
                   P_trans: np.ndarray) -> float:
    """Port of ``Log_Likelihood``: scaled forward recursion."""
    n_data = len(data)
    prob = probabilities[:, 0] * np.asarray(init, dtype=np.float64)
    alpha = prob / prob.sum()
    ll = np.log(prob.sum())
    for i in range(n_data - 1):
        prob = (np.asarray(P_trans).T @ alpha) * probabilities[:, i + 1]
        alpha = prob / prob.sum()
        ll += np.log(prob.sum())
    return float(ll)


def normal_params(g: np.ndarray, y: np.ndarray, y2: np.ndarray, n_levels: int,
                  variant: str, analytic_jac: bool = True) -> np.ndarray:
    """Port of the emission M-step ``normal_params`` (``variant``  is ``vnd`` or
    ``sdmc``; the SDMC version is the hardened re-implementation from the same
    file, used identically for both packages here)."""
    g = np.asarray(g, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)

    if variant == "sdmc":
        g_safe = np.maximum(g, 1e-12)
        init_vals = y / g_safe
        diffs = (init_vals[1:] - init_vals[0]) / np.arange(1, n_levels)
        diffs = diffs[np.isfinite(diffs)]
        step = float(np.max(diffs)) if diffs.size else 5.0
    else:
        init_vals = y / np.where(g == 0, np.nan, g)
        init_vals = np.where(np.isfinite(init_vals), init_vals, 0.0)
        step = (init_vals[-1] - init_vals[0]) / (n_levels - 1)

    if not np.isfinite(step) or step <= 0:
        step = 5.0
    x0 = np.concatenate([[init_vals[0], step], np.full(n_levels, 10.0 / step)])

    def mcosti(x):
        mu = x[0] + np.arange(n_levels) * x[1]
        s2 = x[2:] ** 2
        return float(np.sum((y2 - 2 * mu * y + g * mu**2) * s2 - g * np.log(np.maximum(s2, 1e-300))))

    def mcosti_jac(x):
        mu = x[0] + np.arange(n_levels) * x[1]
        s = x[2:]
        r = g * mu - y
        db = 2.0 * np.sum(r * s**2)
        dh = 2.0 * np.sum(np.arange(n_levels) * r * s**2)
        ds = 2.0 * s * (y2 - 2.0 * mu * y + g * mu**2) - 2.0 * g / np.maximum(s, 1e-300)
        return np.concatenate([[db, dh], ds])

    cons = [
        {"type": "ineq", "fun": lambda x: x[1]},
        {"type": "ineq", "fun": lambda x: x[2:]},
    ]
    res = minimize(mcosti, x0, jac=mcosti_jac if analytic_jac else None,
                   method="SLSQP", constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x


def viterbi(data: np.ndarray, init: np.ndarray, P_trans: np.ndarray,
            mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Port of ``Viterbi_simple``; returns 0-based state indices."""
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    K = len(mu)
    sigma = np.sqrt(var)
    delta = np.zeros((K, n))
    psi = np.zeros((K, max(n - 1, 0)), dtype=np.int64)
    delta[:, 0] = gaussian_densities(data[:1], mu, sigma)[:, 0] * np.asarray(init)
    for i in range(n - 1):
        scores = delta[:, i][:, None] * np.asarray(P_trans)
        best_prev = np.argmax(scores, axis=0)
        best_val = scores[best_prev, np.arange(K)]
        psi[:, i] = best_prev
        delta[:, i + 1] = gaussian_densities(data[i + 1:i + 2], mu, sigma)[:, 0] * best_val
        if delta[:, i + 1].max() == 0:
            last_prob = delta[:, i] * np.asarray(P_trans)[:, K - 1]
            tmp = np.min(mu - data[i + 1])
            delta[:, i + 1] = gaussian_densities(
                data[i + 1:i + 2], mu, np.full(K, abs(tmp) / 35.0))[:, 0] * last_prob.max()
        pot = np.floor(np.log10(delta[:, i + 1].max()))
        delta[:, i + 1] *= 10.0 ** (-pot)
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(delta[:, -1]))
    for i in range(n - 2, -1, -1):
        path[i] = psi[path[i + 1], i]
    return path


def viterbi_bruteforce(data: np.ndarray, init: np.ndarray, P_trans: np.ndarray,
                       mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Reference Viterbi for verification on short traces (log space)."""
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    K = len(mu)
    logp = np.log(gaussian_densities(data, mu, np.sqrt(var)) + 1e-300)
    logP = np.log(np.asarray(P_trans) + 1e-300)
    logdelta = np.log(np.asarray(init) + 1e-300) + logp[:, 0]
    back = np.zeros((K, n), dtype=np.int64)
    for t in range(1, n):
        scores = logdelta[:, None] + logP
        back[:, t] = np.argmax(scores, axis=0)
        logdelta = scores[back[:, t], np.arange(K)] + logp[:, t]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(logdelta))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[path[t], t]
    return path


def binomial_transition(n_channels: int, p_stay_closed: float, p_stay_open: float) -> np.ndarray:
    """Independent-channel reference transition matrix on the sum states."""
    l = n_channels
    T = np.zeros((l + 1, l + 1))
    for i in range(l + 1):
        for j in range(l + 1):
            val = 0.0
            for k in range(max(0, i - j), min(i, l - j) + 1):
                val += (comb(i, k) * comb(l - i, j - i + k)
                        * p_stay_open ** (i - k) * (1 - p_stay_open) ** k
                        * p_stay_closed ** (l - j - k) * (1 - p_stay_closed) ** (j - i + k))
            T[i, j] = val
    return T

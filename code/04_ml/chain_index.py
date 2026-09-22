"""Precomputed index tables that turn the count chain into a differentiable op.

``kinetics.count_transition_matrix`` builds the exact occupancy-count transition
matrix with a Python loop over states, which cannot sit inside a training step.
The same matrix is a polynomial in the 49 entries of the single-channel kernel
``P0``: moving N channels one step means choosing, for each channel, one cell
(i -> j) of ``P0``. So every term of the matrix is

    coefficient * P0[i1,j1] * P0[i2,j2] * ... * P0[iN,jN]

with exactly N factors. Enumerating the multisets of N cells out of 49 therefore
enumerates every term exactly once, and the tables below store, per term, the
source state row, the destination state column, the multinomial coefficient and
the N flat indices into ``P0``. Rebuilding the matrix for a new ``P0`` is then a
gather, a product and a scatter-add -- all differentiable.

Term counts are 49, 1225, 20825, 270725 and 2869685 for N = 1..5 (about 63 MB for
N=5), so the tables are generated once and cached in
``data/derived/chain_index/``.

Usage:
    python code/04_ml/chain_index.py --build        # build and cache N = 1..5
    python code/04_ml/chain_index.py --verify       # check against kinetics.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "04_ml"))

N_STATES = 7
N_CELLS = N_STATES * N_STATES
CACHE = ROOT / "data" / "derived" / "chain_index"
_FACT = np.array([1, 1, 2, 6, 24, 120, 720, 5040], dtype=np.float64)


def sorted_multisets(n_items: int, k: int) -> np.ndarray:
    """All non-decreasing k-tuples over ``range(n_items)``, shape (C(n+k-1,k), k)."""
    if k == 0:
        return np.zeros((1, 0), dtype=np.int16)
    out = np.arange(n_items, dtype=np.int16)[:, None]
    for _ in range(k - 1):
        tail = out[:, -1].astype(np.int64)
        widths = n_items - tail
        idx = np.repeat(np.arange(len(out)), widths)
        starts = np.repeat(np.cumsum(widths) - widths, widths)
        nxt = np.empty((len(idx), out.shape[1] + 1), dtype=np.int16)
        nxt[:, :-1] = out[idx]
        nxt[:, -1] = tail[idx] + (np.arange(len(idx)) - starts)
        out = nxt
    return out


def count_states(N: int) -> np.ndarray:
    """Occupancy vectors summing to N, in the same order as ``kinetics.count_states``."""
    states: list[tuple[int, ...]] = []

    def rec(prefix: list[int], remaining: int) -> None:
        if len(prefix) == N_STATES - 1:
            states.append(tuple(prefix + [remaining]))
            return
        for v in range(remaining + 1):
            rec(prefix + [v], remaining - v)

    rec([], N)
    return np.asarray(states, dtype=np.int16)


def _state_lookup(states: np.ndarray, N: int) -> tuple[np.ndarray, int]:
    """Mixed-radix table mapping an occupancy vector to its row index."""
    base = N + 1
    powers = base ** np.arange(N_STATES, dtype=np.int64)
    lut = np.full(base**N_STATES, -1, dtype=np.int32)
    lut[states.astype(np.int64) @ powers] = np.arange(len(states), dtype=np.int32)
    return lut, base


def build_chain_index(N: int) -> dict[str, np.ndarray]:
    """Term table for the exact N-channel occupancy-count transition matrix."""
    picks = sorted_multisets(N_CELLS, N)
    rows_of, cols_of = picks // N_STATES, picks % N_STATES

    src = np.stack([(rows_of == i).sum(axis=1) for i in range(N_STATES)], axis=1)
    dst = np.stack([(cols_of == j).sum(axis=1) for j in range(N_STATES)], axis=1)

    # multinomial coefficient prod_i n_i! / prod_ij m_ij!; the denominator is the
    # product of the within-run ranks of the sorted picks.
    rank = np.ones(len(picks), dtype=np.float64)
    for u in range(1, N):
        rank *= 1.0 + (picks[:, :u] == picks[:, u : u + 1]).sum(axis=1)
    coef = _FACT[src].prod(axis=1) / rank

    states = count_states(N)
    lut, base = _state_lookup(states, N)
    powers = base ** np.arange(N_STATES, dtype=np.int64)
    rows = lut[src.astype(np.int64) @ powers]
    cols = lut[dst.astype(np.int64) @ powers]
    if (rows < 0).any() or (cols < 0).any():
        raise RuntimeError("state lookup failed")

    order = np.argsort(rows.astype(np.int64) * len(states) + cols, kind="stable")
    log_multinom = np.log(_FACT[N]) - np.log(_FACT[states]).sum(axis=1)
    return {
        "picks": picks[order],
        "rows": rows[order].astype(np.int32),
        "cols": cols[order].astype(np.int32),
        "coef": coef[order].astype(np.float32),
        "states": states.astype(np.int16),
        "log_multinom": log_multinom.astype(np.float32),
    }


def load_chain_index(N: int, cache: pathlib.Path = CACHE) -> dict[str, np.ndarray]:
    path = cache / f"chain_{N}.npz"
    if path.exists():
        with np.load(path) as d:
            return {k: d[k] for k in d.files}
    tables = build_chain_index(N)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **tables)
    return tables


def transition_matrix_numpy(P0: np.ndarray, tables: dict[str, np.ndarray]) -> np.ndarray:
    """Reference (numpy) rebuild, used by the verification path."""
    S = len(tables["states"])
    vals = tables["coef"].astype(np.float64) * P0.reshape(-1)[tables["picks"]].prod(axis=1)
    T = np.zeros(S * S)
    np.add.at(T, tables["rows"].astype(np.int64) * S + tables["cols"], vals)
    return T.reshape(S, S)


def _build(max_n: int) -> None:
    for N in range(1, max_n + 1):
        tables = load_chain_index(N)
        size = sum(v.nbytes for v in tables.values()) / 1e6
        print(f"N={N}: {len(tables['picks']):>9,} terms, {len(tables['states']):>4} states, {size:6.1f} MB")


def _verify(max_n: int) -> None:
    import benchmark as B
    import benchmark_v2 as B2
    import kinetics as K
    from scipy.linalg import expm

    rng = np.random.default_rng(0)
    worst_T = worst_pi = 0.0
    for N in range(1, max_n + 1):
        tables = load_chain_index(N)
        for trial in range(3):
            rates = B2.BASE_OFF * np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=12))
            P0 = expm(B.DT * B2.rate_table(rates))
            T_ref = K.count_transition_matrix(N, P0)
            T_new = transition_matrix_numpy(P0, tables)
            dT = float(np.abs(T_ref - T_new).max())

            pi_single = B.stationary(P0)
            pi_ref = K.count_stationary(N, P0)
            pi_new = np.exp(tables["log_multinom"] + tables["states"] @ np.log(pi_single))
            dpi = float(np.abs(pi_ref - pi_new).max())

            worst_T, worst_pi = max(worst_T, dT), max(worst_pi, dpi)
            if trial == 0:
                rowsum = float(np.abs(T_new.sum(axis=1) - 1).max())
                print(f"N={N}: max|dT|={dT:.2e}  max|dpi|={dpi:.2e}  max|rowsum-1|={rowsum:.2e}")
    print(f"\nworst over all trials: transition {worst_T:.2e}, stationary {worst_pi:.2e}")
    print("PASS" if max(worst_T, worst_pi) < 1e-6 else "FAIL")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--max-n", type=int, default=5)
    args = parser.parse_args()
    if args.build or not args.verify:
        _build(args.max_n)
    if args.verify:
        _verify(args.max_n)


if __name__ == "__main__":
    main()

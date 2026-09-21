"""Rate-randomized benchmark (synth_v2): N, full state counts, and Markov parameters.

Extends the frozen synth_v1 benchmark (``code/04_ml/benchmark.py``) with:

  - a **random rate table per group** (a group = traces recorded under the same
    kinetics), so estimating Markov parameters is a real task, not a constant;
  - the full 7-state label ``r`` per channel per time, so a model can be trained
    to output the per-state channel counts a..g (sum = N);
  - the 12 off-diagonal rates of the CFTR kinetic graph used to generate each
    group, as the target for a Markov-parameter head.

Artifacts land in ``data/derived/synth_v2/`` (``data/raw`` is never touched).

Usage:
    python code/04_ml/benchmark_v2.py --generate
    python code/04_ml/benchmark_v2.py --mini --generate   # tiny, for smoke tests
    python code/04_ml/benchmark_v2.py --info
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.linalg import expm
from scipy.signal import lfilter

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "04_ml"))

import benchmark as B  # noqa: E402

DEST = ROOT / "data" / "derived" / "synth_v2"

# The 12 allowed transitions of the fixed 7-state CFTR graph, in a fixed order.
RATE_INDEX = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2),
              (3, 4), (4, 5), (5, 4), (5, 6), (6, 5), (6, 0)]
RATE_NAMES = [f"{i}->{j}" for i, j in RATE_INDEX]

# Effective base rates (per second). (0,1) is 10/s: ion_channel.py overwrites the
# literal 9 with C1aExitProb/dt = 10.
BASE_OFF = np.array([10.0, 5.0, 7.7, 5.8, 4.9, 10.0,
                     7.1, 3.0, 7.0, 6.0, 12.8, 1.7], dtype=np.float64)

P_OPEN_RANGE = (0.10, 0.90)


def rate_table(values) -> np.ndarray:
    """Build a 7x7 rate matrix (rows sum to 0) from the 12 off-diagonal rates."""
    R = np.zeros((7, 7), dtype=np.float64)
    for v, (i, j) in zip(np.atleast_1d(values), RATE_INDEX):
        R[i, j] = v
    R[np.diag_indices(7)] = -R.sum(axis=1)
    return R


def sample_rate_tables(
    rng: np.random.Generator,
    n_groups: int,
    scale_range=(0.5, 2.0),
    p_open_range=P_OPEN_RANGE,
    max_tries: int = 100,
) -> np.ndarray:
    """Draw one rate table per group: base rates x log-uniform(factor)."""
    lo, hi = np.log(float(scale_range[0])), np.log(float(scale_range[1]))
    tables = np.empty((n_groups, len(RATE_INDEX)), dtype=np.float64)
    for g in range(n_groups):
        for _ in range(max_tries):
            vals = BASE_OFF * np.exp(rng.uniform(lo, hi, size=len(RATE_INDEX)))
            pi = B.stationary(expm(B.DT * rate_table(vals)))
            p_open = float(pi[B.STATEMAP == 1].sum())
            if p_open_range[0] <= p_open <= p_open_range[1]:
                tables[g] = vals
                break
        else:
            tables[g] = BASE_OFF
    return tables


def sample_states_channels(
    rng: np.random.Generator,
    P0_ch: np.ndarray,
    pi_ch: np.ndarray,
    n_samples: int,
    P0_ch_second: np.ndarray | None = None,
) -> np.ndarray:
    """Sample 7-state trajectories for many channels at once.

    ``P0_ch`` and ``pi_ch`` are per-channel one-step matrices and stationary
    distributions (channels of the same group repeat the same matrix). If
    ``P0_ch_second`` is given, the kernel switches at the midpoint (drift).
    """
    n = P0_ch.shape[0]
    r = np.empty((n, n_samples), dtype=np.int8)
    u0 = rng.random(n)
    r[:, 0] = (u0[:, None] > np.cumsum(pi_ch, axis=1)).sum(axis=1).astype(np.int8)

    cdf = np.cumsum(P0_ch, axis=2)
    switch = n_samples // 2
    rows = np.arange(n)
    for t in range(1, n_samples):
        if P0_ch_second is not None and t == switch:
            cdf = np.cumsum(P0_ch_second, axis=2)
        u = rng.random(n)
        r[:, t] = (u[:, None] < cdf[rows, r[:, t - 1]]).argmax(axis=1).astype(np.int8)
    return r


def generate_grouped_split(
    seed: int,
    n_groups: int,
    traces_per_group: int,
    channel_choices=(1, 2, 3, 4, 5),
    n_samples: int = 1000,
    scales: tuple[float, ...] = (1.0,),
    noise_scale_range: tuple[float, float] | None = None,
    noise_kind: str = "gh",
    ar_rho: float = 0.0,
    rate_scale_range: tuple[float, float] = (0.5, 2.0),
    rate_drift_mult: float | None = None,
) -> dict[str, np.ndarray]:
    """Generate one split: groups of traces sharing a random rate table.

    Returns ``X`` (random per-trace noise scale) or ``X_s<scale>`` (fixed scales),
    plus ``y`` (open count), ``r`` (7-state index per channel, padded with -1),
    ``N``, ``group`` and ``R`` (the 12 rates of each group).
    """
    rng = np.random.default_rng(seed)
    tables = sample_rate_tables(rng, n_groups, rate_scale_range)
    P0_g = np.stack([expm(B.DT * rate_table(t)) for t in tables])
    pi_g = np.stack([B.stationary(P0) for P0 in P0_g])

    N = rng.choice(np.asarray(channel_choices), size=(n_groups, traces_per_group)).astype(np.int8)
    n_traces = n_groups * traces_per_group
    group = np.repeat(np.arange(n_groups, dtype=np.int16), traces_per_group)
    N_flat = N.reshape(-1)
    if int(N_flat.sum()) == 0:
        raise ValueError("no channels sampled")

    ch_trace = np.repeat(np.arange(n_traces), N_flat)
    ch_group = group[ch_trace]
    P0_ch = P0_g[ch_group]
    pi_ch = pi_g[ch_group]
    P0_ch2 = None
    if rate_drift_mult is not None:
        tables2 = tables * float(rate_drift_mult)
        P0_g2 = np.stack([expm(B.DT * rate_table(t)) for t in tables2])
        P0_ch2 = P0_g2[ch_group]

    r_ch = sample_states_channels(rng, P0_ch, pi_ch, n_samples, P0_ch2)
    c = B.STATEMAP[r_ch]
    obs = B._channel_observation(c, rng, noise_kind=noise_kind, ar_rho=ar_rho)
    level = np.where(c == 1, B.OPEN_GH["loc"], B.CLOSE_GH["loc"])
    dev = obs - level

    starts = np.concatenate([[0], np.cumsum(N_flat)]).astype(int)[:-1]
    level_sum = np.add.reduceat(level, starts, axis=0)
    dev_sum = np.add.reduceat(dev, starts, axis=0)

    max_ch = int(max(channel_choices))
    r_pad = np.full((n_traces, max_ch, n_samples), -1, dtype=np.int8)
    for i in range(n_traces):
        n = int(N_flat[i])
        r_pad[i, :n] = r_ch[starts[i]:starts[i] + n]
    y = (B.STATEMAP[r_pad] == 1).sum(axis=1).astype(np.int8)

    result = {
        "N": N_flat,
        "y": y,
        "r": r_pad,
        "group": group,
        "R": tables.astype(np.float32),
    }
    if noise_scale_range is not None:
        s = rng.uniform(float(noise_scale_range[0]), float(noise_scale_range[1]), size=(n_traces, 1))
        result["trace_scale"] = s[:, 0].astype(np.float32)
        result["X"] = (level_sum + dev_sum * s).astype(np.float32)
    for sc in scales:
        result[f"X_s{sc:g}"] = (level_sum + dev_sum * float(sc)).astype(np.float32)
    return result


def lowpass_r(data: dict[str, np.ndarray], factor: int = 10, window: int = 15) -> dict[str, np.ndarray]:
    """Moving-average + decimation, keeping y, r, N, group, R aligned in time."""
    X = data["X_s1"]
    kernel = np.ones(window) / window
    X_f = np.stack([np.convolve(row, kernel, mode="same") for row in X])
    keep = np.arange(0, X.shape[1], factor)[: (X.shape[1] // factor)]
    return {
        "X": X_f[:, keep].astype(np.float32),
        "y": data["y"][:, keep],
        "r": data["r"][:, :, keep],
        "N": data["N"],
        "group": data["group"],
        "R": data["R"],
    }


def _meta(split: str, seed: int, **extra) -> dict:
    meta = {
        "split": split,
        "seed": seed,
        "rate_index": RATE_NAMES,
        "base_rates": BASE_OFF.tolist(),
        "rate_scale_range": [0.5, 2.0],
        "p_open_range": list(P_OPEN_RANGE),
    }
    meta.update(extra)
    return meta


def generate_all(dest: pathlib.Path = DEST, mini: bool = False, seed_shift: int = 0) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    seeds = {"train": 9101, "val": 9202, "test": 9303, "extrap": 9404, "mismatch": 9505}
    seeds = {k: v + seed_shift for k, v in seeds.items()}

    if mini:
        g_tr, t_tr, g_ev, t_ev, T = 8, 4, 4, 4, 1000
    else:
        g_tr, t_tr, g_ev, t_ev, T = 512, 6, 64, 6, 1000

    train = generate_grouped_split(
        seeds["train"], g_tr, t_tr, n_samples=T, scales=(), noise_scale_range=(1.0, 4.0)
    )
    B.save_npz(dest / "train.npz", train, _meta("train", seeds["train"], groups=g_tr, traces_per_group=t_tr))

    val = generate_grouped_split(seeds["val"], g_ev, t_ev, n_samples=T)
    B.save_npz(dest / "val.npz", val, _meta("val", seeds["val"], groups=g_ev, traces_per_group=t_ev))

    test = generate_grouped_split(seeds["test"], g_ev, t_ev, n_samples=T, scales=(1.0, 2.0, 4.0))
    B.save_npz(dest / "test.npz", test, _meta("test", seeds["test"], groups=g_ev, traces_per_group=t_ev))

    extrap = generate_grouped_split(
        seeds["extrap"], g_ev, t_ev, n_samples=T, channel_choices=(4, 5), scales=(1.0, 2.0)
    )
    B.save_npz(dest / "extrap.npz", extrap, _meta("extrap", seeds["extrap"], groups=g_ev, traces_per_group=t_ev))

    mm = generate_grouped_split(seeds["mismatch"], g_ev, t_ev, n_samples=T, channel_choices=(1, 2, 3))
    B.save_npz(dest / "mismatch_base.npz", mm, _meta("mismatch_base", seeds["mismatch"]))

    mm_low = lowpass_r(mm)
    B.save_npz(dest / "mismatch_lowpass.npz", mm_low,
               _meta("mismatch_lowpass", seeds["mismatch"], factor=10, window=15))

    mm_gauss = generate_grouped_split(
        seeds["mismatch"], g_ev, t_ev, n_samples=T, channel_choices=(1, 2, 3), noise_kind="gaussian"
    )
    B.save_npz(dest / "mismatch_gaussian.npz", mm_gauss, _meta("mismatch_gaussian", seeds["mismatch"]))

    mm_corr = generate_grouped_split(
        seeds["mismatch"], g_ev, t_ev, n_samples=T, channel_choices=(1, 2, 3), ar_rho=0.9
    )
    B.save_npz(dest / "mismatch_corr.npz", mm_corr, _meta("mismatch_corr", seeds["mismatch"], ar_rho=0.9))

    mm_drift = generate_grouped_split(
        seeds["mismatch"], g_ev, t_ev, n_samples=T, channel_choices=(1, 2, 3), rate_drift_mult=3.0
    )
    B.save_npz(dest / "mismatch_drift.npz", mm_drift, _meta("mismatch_drift", seeds["mismatch"], rate_drift_mult=3.0))


def generate_extra_train(dest: pathlib.Path = DEST, n_groups: int = 1024,
                         traces_per_group: int = 6, seed: int = 9111) -> None:
    """Additional training groups only; val/test/extrap/mismatch stay frozen.

    The channel-count head overfits on 512 groups (train 0.98 vs test 0.88), and
    more simulated groups are the cheapest fix available.
    """
    dest.mkdir(parents=True, exist_ok=True)
    data = generate_grouped_split(seed, n_groups, traces_per_group, n_samples=1000,
                                  scales=(), noise_scale_range=(1.0, 4.0))
    B.save_npz(dest / "train_extra.npz", data,
               _meta("train_extra", seed, groups=n_groups, traces_per_group=traces_per_group))


def augment_trace(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add real-world mess to one trace: wander, trend, colored noise, filtering.

    Labels (gating) are unchanged; only the observation is degraded.
    """
    x = np.asarray(x, dtype=np.float64)
    if rng.random() < 0.2:
        return x.astype(np.float32)
    s = x.std()
    n = len(x)

    walk = np.cumsum(rng.standard_normal(n))
    walk = walk / (np.abs(walk).max() + 1e-9)
    walk = np.convolve(walk, np.ones(51) / 51, mode="same")
    x = x + rng.uniform(0.0, 0.6) * s * walk

    trend = np.linspace(-0.5, 0.5, n)
    x = x + rng.uniform(-0.5, 0.5) * s * trend

    rho = rng.uniform(0.0, 0.9)
    sigma = rng.uniform(0.0, 0.4) * s
    if sigma > 0:
        eps = rng.standard_normal(n) * sigma * np.sqrt(1 - rho**2)
        x = x + lfilter([1.0], [1.0, -rho], eps)

    w = int(rng.integers(1, 6))
    if w > 1:
        x = np.convolve(x, np.ones(w) / w, mode="same")
    return x.astype(np.float32)


def augment_split(data: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = dict(data)
    for key in [k for k in data if k.startswith("X")]:
        X = data[key].copy()
        for i in range(len(X)):
            X[i] = augment_trace(X[i], rng)
        out[key] = X
    return out


def generate_augmented(dest: pathlib.Path = DEST, mini: bool = False) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    base_train = load_split("train", dest)
    base_val = load_split("val", dest)
    train = augment_split(base_train, seed=0 if mini else 7707)
    val = augment_split(base_val, seed=1 if mini else 8808)
    B.save_npz(dest / "train_aug.npz", train, _meta("train_aug", 7707))
    B.save_npz(dest / "val_aug.npz", val, _meta("val_aug", 8808))


def load_split(name: str, dest: pathlib.Path = DEST) -> dict[str, np.ndarray]:
    with np.load(dest / f"{name}.npz") as data:
        return {k: data[k] for k in data.files if k != "meta_json"}


def _info() -> None:
    if not DEST.exists():
        print("no synth_v2 yet (run --generate)")
        return
    print(f"dest: {DEST}")
    for f in sorted(DEST.glob("*.npz")):
        with np.load(f) as d:
            xkeys = [k for k in d.files if k.startswith("X")]
            shape = d[xkeys[0]].shape
            print(f"{f.name:26s} X{shape} groups={len(np.unique(d['group']))} "
                  f"keys={sorted(k for k in d.files if k != 'meta_json')}")
    print("\nchecks on train.npz:")
    tr = load_split("train")
    r, N, y, X, R = tr["r"], tr["N"], tr["y"], tr["X"], tr["R"]
    counts = np.stack([(r == s).sum(axis=1) for s in range(7)], axis=1)  # (traces, 7, T)
    total = counts.sum(axis=1)
    sum_ok = bool((total == N[:, None]).all())
    open_states = np.flatnonzero(B.STATEMAP == 1)  # O1, O2 = indices 3, 4
    open_ok = bool((counts[:, open_states].sum(axis=1) == y).all())
    print(f"  sum a..g == N   : {sum_ok}   (max deviation {int(np.abs(total - N[:, None]).max())})")
    print(f"  open count == y : {open_ok}")
    print(f"  rates positive  : {bool((R > 0).all())}, groups: {len(R)}")
    print(f"  X range         : {float(X.min()):.3f} .. {float(X.max()):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--extra-train", type=int, default=0,
                        help="generate this many additional training groups (train_extra.npz)")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--info", action="store_true")
    args = parser.parse_args()
    if args.generate:
        generate_all(mini=args.mini)
    if args.augment:
        generate_augmented(mini=args.mini)
    if args.extra_train:
        generate_extra_train(n_groups=args.extra_train)
    if args.info or not (args.generate or args.augment or args.extra_train):
        _info()


if __name__ == "__main__":
    main()

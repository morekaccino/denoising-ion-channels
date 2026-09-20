"""Frozen synthetic benchmark for multi-channel CFTR patch-clamp state estimation.

Reproduces the generative model of ``source/ion_channel.py`` (7-state CFTR kinetic
chain, state-dependent generalized-hyperbolic noise, channels summed by
``source/patch_clamp.py``) with full RNG control and independent train/val/test
splits, plus evaluation metrics that the thesis never reported (per-channel-count
breakdown, transition-window accuracy, calibration, N estimation).

Artifacts land in ``data/derived/synth_v1/`` (``data/raw`` is never touched).

Usage:
    python code/04_ml/benchmark.py --generate
    python code/04_ml/benchmark.py --info
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import scipy.stats as st
from scipy.linalg import expm

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "derived" / "synth_v1"

DT = 0.01
N_SAMPLES = 1000
C1A_EXIT_PROB = 0.1
STATEMAP = np.array([0, 0, 0, 1, 1, 0, 0])

OPEN_GH = {
    "p": 1.9865096704542702,
    "a": 0.00199882860659155,
    "b": -0.0005556910258636614,
    "loc": 1.4,
    "scale": 0.0002071081765532621,
}
CLOSE_GH = {
    "p": 3.6662368635821796,
    "a": 0.5836326667155399,
    "b": 0.16863569591265098,
    "loc": 0.58,
    "scale": 0.024246693602449965,
}

BASE_RATES = np.array(
    [
        [-9.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [5.0, -12.7, 7.7, 0.0, 0.0, 0.0, 0.0],
        [0.0, 5.8, -10.7, 4.9, 0.0, 0.0, 0.0],
        [0.0, 0.0, 10.0, -17.1, 7.1, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -3.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 7.0, -13.0, 6.0],
        [1.7, 0.0, 0.0, 0.0, 0.0, 12.8, -14.5],
    ]
)


def rate_matrix(rate_scale: float = 1.0) -> np.ndarray:
    R = BASE_RATES.copy()
    if rate_scale != 1.0:
        off = ~np.eye(7, dtype=bool)
        R[off] *= rate_scale
        R[np.diag_indices(7)] = 0.0
        R[np.diag_indices(7)] = -R.sum(axis=1)
    R[0, 0] = -C1A_EXIT_PROB / DT
    R[0, 1] = C1A_EXIT_PROB / DT
    return R


def p0_matrix(rate_scale: float = 1.0) -> np.ndarray:
    return expm(DT * rate_matrix(rate_scale))


def stationary(P: np.ndarray) -> np.ndarray:
    u, v = np.linalg.eig(P.T)
    i = int(np.argmin(np.abs(u - 1.0)))
    pi = np.real(v[:, i])
    pi = pi / pi.sum()
    return np.clip(pi, 0.0, None)


def sample_states(n_chains: int, n_samples: int, rng: np.random.Generator, P: np.ndarray) -> np.ndarray:
    pi = stationary(P)
    r = np.empty((n_chains, n_samples), dtype=np.int8)
    r[:, 0] = rng.choice(7, size=n_chains, p=pi)
    cdf = np.cumsum(P, axis=1)
    for t in range(1, n_samples):
        u = rng.random(n_chains)
        r[:, t] = (u[:, None] < cdf[r[:, t - 1]]).argmax(axis=1)
    return r


def sample_states_drift(
    n_chains: int, n_samples: int, rng: np.random.Generator, P_first: np.ndarray, P_second: np.ndarray
) -> np.ndarray:
    split = n_samples // 2
    first = sample_states(n_chains, split, rng, P_first)
    r = np.empty((n_chains, n_samples), dtype=np.int8)
    r[:, :split] = first
    cdf = np.cumsum(P_second, axis=1)
    for t in range(split, n_samples):
        u = rng.random(n_chains)
        r[:, t] = (u[:, None] < cdf[r[:, t - 1]]).argmax(axis=1)
    return r


def _standardized_noise(
    shape: tuple[int, ...],
    rng: np.random.Generator,
    kind: str,
    params: dict,
    ar_rho: float = 0.0,
) -> np.ndarray:
    if kind == "gh":
        z = st.genhyperbolic.rvs(
            params["p"], params["a"], params["b"], loc=0.0, scale=1.0, size=shape, random_state=rng
        )
    elif kind == "gaussian":
        std = st.genhyperbolic.std(params["p"], params["a"], params["b"], loc=0.0, scale=1.0)
        z = rng.standard_normal(shape) * std
    else:
        raise ValueError(f"unknown noise kind {kind!r}")
    if ar_rho > 0.0:
        eps = np.sqrt(1.0 - ar_rho**2)
        for t in range(1, shape[-1]):
            z[..., t] = ar_rho * z[..., t - 1] + eps * z[..., t]
    return np.asarray(z, dtype=np.float64)


def _channel_observation(
    open_indicator: np.ndarray,
    rng: np.random.Generator,
    noise_kind: str = "gh",
    ar_rho: float = 0.0,
) -> np.ndarray:
    zo = _standardized_noise(open_indicator.shape, rng, noise_kind, OPEN_GH, ar_rho)
    zc = _standardized_noise(open_indicator.shape, rng, noise_kind, CLOSE_GH, ar_rho)
    is_open = open_indicator == 1
    level = np.where(is_open, OPEN_GH["loc"], CLOSE_GH["loc"])
    spread = np.where(is_open, OPEN_GH["scale"], CLOSE_GH["scale"])
    z = np.where(is_open, zo, zc)
    return level + spread * z


def generate_split(
    seed: int,
    n_traces: int,
    channel_choices: tuple[int, ...] = (1, 2, 3),
    n_samples: int = N_SAMPLES,
    scales: tuple[float, ...] = (1.0,),
    noise_kind: str = "gh",
    ar_rho: float = 0.0,
    rate_scale: float = 1.0,
    drift_rate_scale: float | None = None,
    scale_range: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    P = p0_matrix(rate_scale)
    P2 = p0_matrix(drift_rate_scale) if drift_rate_scale is not None else None
    N = rng.choice(np.asarray(channel_choices), size=n_traces).astype(np.int8)
    max_ch = int(max(channel_choices))

    out = {scope: np.empty((n_traces, n_samples), dtype=np.float32) for scope in scales}
    y = np.empty((n_traces, n_samples), dtype=np.int8)
    r_padded = np.full((n_traces, max_ch, n_samples), -1, dtype=np.int8)
    trace_scale = np.ones(n_traces, dtype=np.float32)

    for n in sorted({int(v) for v in N}):
        idx = np.flatnonzero(N == n)
        m = len(idx)
        if P2 is None:
            r = sample_states(m * n, n_samples, rng, P).reshape(m, n, n_samples)
        else:
            r = sample_states_drift(m * n, n_samples, rng, P, P2).reshape(m, n, n_samples)
        r_padded[idx, :n] = r
        c = STATEMAP[r]
        y[idx] = c.sum(axis=1)
        chan = _channel_observation(c, rng, noise_kind=noise_kind, ar_rho=ar_rho)
        level = np.where(c == 1, OPEN_GH["loc"], CLOSE_GH["loc"])
        for s in scales:
            out[s][idx] = (level + (chan - level) * s).sum(axis=1)
        if scale_range is not None:
            s = rng.uniform(scale_range[0], scale_range[1], size=(m, 1, 1)).astype(np.float32)
            trace_scale[idx] = s[:, 0, 0]
            out.setdefault("X_rand", np.empty((n_traces, n_samples), dtype=np.float32))
            out["X_rand"][idx] = (level + (chan - level) * s).sum(axis=1)

    result = {"N": N, "y": y, "r": r_padded}
    if scale_range is not None:
        result["trace_scale"] = trace_scale
        result["X_rand"] = out["X_rand"]
    for s in scales:
        result[f"X_s{s:g}"] = out[s]
    return result


def lowpass_decimate(
    data: dict[str, np.ndarray], factor: int = 10, window: int = 15
) -> dict[str, np.ndarray]:
    X = data["X_s1"]
    kernel = np.ones(window) / window
    X_f = np.stack([np.convolve(row, kernel, mode="same") for row in X])
    keep = np.arange(0, X.shape[1], factor)[: (X.shape[1] // factor)]
    return {
        "X": X_f[:, keep].astype(np.float32),
        "y": data["y"][:, keep],
        "N": data["N"],
        "factor": np.int8(factor),
        "window": np.int8(window),
    }


def save_npz(path: pathlib.Path, arrays: dict[str, np.ndarray], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **payload)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def generate_all(dest: pathlib.Path = DEST) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    rng_seeds = {"train": 1101, "val": 2202, "test": 3303, "extrap": 4404, "mismatch": 5505}

    train = generate_split(rng_seeds["train"], 1500, (1, 2, 3), scales=(1.0,))
    save_npz(dest / "train.npz", train, {"split": "train", "seed": rng_seeds["train"], "scales": [1.0]})

    val = generate_split(rng_seeds["val"], 300, (1, 2, 3), scales=(1.0,))
    save_npz(dest / "val.npz", val, {"split": "val", "seed": rng_seeds["val"], "scales": [1.0]})

    test = generate_split(rng_seeds["test"], 600, (1, 2, 3), scales=(1.0, 2.0, 4.0))
    save_npz(dest / "test.npz", test, {"split": "test", "seed": rng_seeds["test"], "scales": [1.0, 2.0, 4.0]})

    extrap = generate_split(rng_seeds["extrap"], 400, (4, 5), scales=(1.0, 2.0))
    save_npz(dest / "extrap.npz", extrap, {"split": "extrap", "seed": rng_seeds["extrap"], "scales": [1.0, 2.0]})

    wide = generate_split(
        6607, 3000, (1, 2, 3, 4, 5), scale_range=(1.0, 4.0)
    )
    save_npz(
        dest / "train_wide.npz",
        {"X": wide["X_rand"], "y": wide["y"], "N": wide["N"], "trace_scale": wide["trace_scale"]},
        {"split": "train_wide", "seed": 6607, "scale_range": [1.0, 4.0]},
    )

    mm = generate_split(rng_seeds["mismatch"], 200, (1, 2, 3), scales=(1.0,))
    save_npz(
        dest / "mismatch_lowpass.npz",
        lowpass_decimate(mm),
        {"split": "mismatch_lowpass", "source_seed": rng_seeds["mismatch"], "factor": 10, "window": 15},
    )

    mm_gauss = generate_split(rng_seeds["mismatch"], 200, (1, 2, 3), scales=(1.0,), noise_kind="gaussian")
    save_npz(
        dest / "mismatch_gaussian.npz",
        {"X": mm_gauss["X_s1"], "y": mm_gauss["y"], "N": mm_gauss["N"]},
        {"split": "mismatch_gaussian", "source_seed": rng_seeds["mismatch"], "noise_kind": "gaussian"},
    )

    mm_corr = generate_split(rng_seeds["mismatch"], 200, (1, 2, 3), scales=(1.0,), ar_rho=0.9)
    save_npz(
        dest / "mismatch_corr.npz",
        {"X": mm_corr["X_s1"], "y": mm_corr["y"], "N": mm_corr["N"]},
        {"split": "mismatch_corr", "source_seed": rng_seeds["mismatch"], "ar_rho": 0.9},
    )

    mm_drift = generate_split(rng_seeds["mismatch"], 200, (1, 2, 3), scales=(1.0,), drift_rate_scale=3.0)
    save_npz(
        dest / "mismatch_drift.npz",
        {"X": mm_drift["X_s1"], "y": mm_drift["y"], "N": mm_drift["N"]},
        {"split": "mismatch_drift", "source_seed": rng_seeds["mismatch"], "drift_rate_scale": 3.0},
    )


def load_split(name: str, dest: pathlib.Path = DEST) -> dict[str, np.ndarray]:
    with np.load(dest / f"{name}.npz") as data:
        return {k: data[k] for k in data.files if k != "meta_json"}


def histogram_features(X: np.ndarray, bins: int = 100) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, bins + 1)
    feats = np.empty((len(X), bins), dtype=np.float32)
    for i, row in enumerate(X):
        norm = (row - row.min()) / (row.max() - row.min())
        feats[i] = np.histogram(norm, bins=edges)[0]
    return feats


def window_dataset(
    X: np.ndarray, y: np.ndarray, n_points: int = 50, include_center: bool = False, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    windows, targets = [], []
    for row, labels in zip(X, y):
        for t in range(n_points, len(row) - n_points):
            lo = t - n_points
            hi = t + n_points + (1 if include_center else 0)
            w = row[lo:hi] if include_center else np.concatenate([row[lo:t], row[t + 1 : hi + 1]])
            windows.append(w)
            targets.append(labels[t])
        if stride > 1:
            windows = windows[::stride]
            targets = targets[::stride]
            break
    return np.asarray(windows, dtype=np.float32), np.asarray(targets, dtype=np.int64)


def transition_mask(y: np.ndarray, w: int = 5) -> np.ndarray:
    y2 = np.atleast_2d(y)
    mask = np.zeros(y2.shape, dtype=bool)
    for row, mrow in zip(y2, mask):
        changes = np.flatnonzero(np.diff(row) != 0) + 1
        for t in changes:
            mrow[max(0, t - w) : min(len(row), t + w + 1)] = True
    return mask if np.ndim(y) == 2 else mask[0]


def open_count_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> float:
    from sklearn.metrics import f1_score

    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    return float(
        f1_score(y_true, y_pred, labels=np.arange(n_classes), average="macro", zero_division=0)
    )


def per_n_breakdown(y_true: np.ndarray, y_pred: np.ndarray, N: np.ndarray) -> dict[int, dict[str, float]]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    N = np.asarray(N)
    if N.shape != y_true.shape:
        if y_true.ndim == 2 and N.ndim == 1:
            N = np.broadcast_to(N[:, None], y_true.shape)
        else:
            raise ValueError(f"cannot broadcast N {N.shape} to labels {y_true.shape}")
    rows = {}
    for n in np.unique(N):
        mask = N == n
        rows[int(n)] = {
            "traces": int(mask.sum()),
            "accuracy": open_count_accuracy(y_true[mask], y_pred[mask]),
            "macro_f1": macro_f1(y_true[mask], y_pred[mask]),
        }
    return rows


def transition_metrics(y_true: np.ndarray, y_pred: np.ndarray, w: int = 5) -> dict[str, float]:
    y_true = np.atleast_2d(y_true)
    y_pred = np.atleast_2d(y_pred)
    mask = transition_mask(y_true, w)
    in_acc = float((y_true[mask] == y_pred[mask]).mean()) if mask.any() else float("nan")
    out_acc = float((y_true[~mask] == y_pred[~mask]).mean()) if (~mask).any() else float("nan")

    detected, errors = 0, []
    for row_t, row_p in zip(y_true, y_pred):
        changes = np.flatnonzero(np.diff(row_t) != 0) + 1
        pred_changes = np.flatnonzero(np.diff(row_p) != 0) + 1
        for t in changes:
            near = pred_changes[np.abs(pred_changes - t) <= w]
            if len(near):
                detected += 1
                errors.append(abs(int(near[0]) - int(t)))
    return {
        "transition_accuracy": in_acc,
        "non_transition_accuracy": out_acc,
        "transition_detection_rate": detected / max(1, sum(len(np.flatnonzero(np.diff(r) != 0)) for r in y_true)),
        "mean_timing_error": float(np.mean(errors)) if errors else float("nan"),
    }


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def n_estimation_metrics(N_true: np.ndarray, N_pred: np.ndarray) -> dict[str, float]:
    N_true, N_pred = np.asarray(N_true), np.asarray(N_pred)
    return {
        "n_accuracy": float((N_true == N_pred).mean()),
        "n_mae": float(np.abs(N_true.astype(float) - N_pred.astype(float)).mean()),
    }


def summarize(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray | None = None,
    N_true: np.ndarray | None = None,
    N_pred: np.ndarray | None = None,
) -> dict:
    result = {
        "accuracy": open_count_accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        **transition_metrics(y_true, y_pred),
        "per_n": per_n_breakdown(y_true, y_pred, N_true) if N_true is not None else {},
    }
    if probs is not None:
        result["ece"] = expected_calibration_error(probs, np.asarray(y_true).ravel())
    if N_true is not None and N_pred is not None:
        result.update(n_estimation_metrics(N_true, N_pred))
    return result


def _info() -> None:
    P = p0_matrix()
    pi = stationary(P)
    p_open = float(pi[STATEMAP == 1].sum())
    print(f"dest          : {DEST}")
    print(f"p_open        : {p_open:.4f}")
    print(f"open-level    : {OPEN_GH['loc']:.3f} (scale {OPEN_GH['scale']:.2e})")
    print(f"closed-level  : {CLOSE_GH['loc']:.3f} (scale {CLOSE_GH['scale']:.4f})")
    if DEST.exists():
        for f in sorted(DEST.glob("*.npz")):
            with np.load(f) as d:
                shape = d["X_s1"].shape if "X_s1" in d.files else d["X"].shape
            print(f"{f.name:28s} {shape}")
    else:
        print("no frozen dataset yet (run --generate)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--info", action="store_true")
    args = parser.parse_args()
    if args.generate:
        generate_all()
    if args.info or not args.generate:
        _info()


if __name__ == "__main__":
    main()

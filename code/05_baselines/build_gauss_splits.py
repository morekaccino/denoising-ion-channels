"""Build Gaussian-noise versions of the synth_v2 training splits.

The two-state HMM baselines (VND, SD-HMM) and the Moffett factor graph assume
Gaussian emissions, while the frozen benchmark uses the generalized-hyperbolic
noise fitted to the real recordings. For the noise-matched control in
``assumption_controls.py`` we need a KI-HMM v5a trained under the *same*
Gaussian emission model as the baselines, so this script rewrites the training
signals with per-channel Gaussian noise of the measured state-dependent
standard deviation (closed 0.128, open 0.232), keeping labels, groups, rate
tables and per-trace noise scales identical.

Outputs (gitignored, rebuildable):
  data/derived/synth_v2/{train_gauss,train_extra_gauss,val_gauss}.npz

Usage:
  python code/05_baselines/build_gauss_splits.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SYNTH = ROOT / "data" / "derived" / "synth_v2"
SIGMA_CLOSED, SIGMA_OPEN = 0.1280, 0.2317
LEVEL_CLOSED, LEVEL_OPEN = 0.58, 1.4
OPEN_STATES = (3, 4)
SEED = 20260926


def convert(name: str, out: str, rng: np.random.Generator) -> None:
    with np.load(SYNTH / f"{name}.npz") as d:
        N, y, r = d["N"], d["y"], d["r"]
        level = LEVEL_CLOSED * N[:, None] + (LEVEL_OPEN - LEVEL_CLOSED) * y
        sigma = np.where(np.isin(r, OPEN_STATES), SIGMA_OPEN, SIGMA_CLOSED)
        valid = (r >= 0).astype(np.float64)
        dev = (rng.normal(0.0, sigma) * valid).sum(axis=1)
        scale = (d["trace_scale"][:, None] if "trace_scale" in d.files
                 else np.float32(1.0))
        X = (level + dev * scale).astype(np.float32)
        keys = {k: d[k] for k in d.files if k != "meta_json"}
    xkey = "X" if "X" in keys else "X_s1"
    keys[xkey] = X
    np.savez_compressed(SYNTH / out, **keys)
    print(f"wrote {SYNTH / out} ({X.shape[0]} traces, {X.shape[1]} samples)")


def main() -> None:
    rng = np.random.default_rng(SEED)
    for name, out in (("train", "train_gauss"),
                      ("train_extra", "train_extra_gauss"),
                      ("val", "val_gauss")):
        if (SYNTH / f"{name}.npz").exists():
            convert(name, out, rng)
        else:
            print(f"skip {name} (missing; build with benchmark_v2 first)",
                  file=sys.stderr)


if __name__ == "__main__":
    main()

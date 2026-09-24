"""Re-run KI-HMM v5a once per noise level and store per-trace predictions.

The published v5 metrics live in ``presentation_v5_metrics.json`` (noise x1) and
``kihmm_v5_infer.json`` (aggregates for x2/x4). The baseline comparisons need
per-trace v5 predictions on the *same* traces, in particular for the N=1
subset used by the single-channel Moffett baseline. This script mirrors the
published presentation stage exactly (evidence-based N, refined rates and
scale, posterior counts) and saves the per-trace counts for all three splits.

Usage:
  python code/05_baselines/v5_reference.py --splits test_x1,test_x2,test_x4
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "04_ml"))

import benchmark as B  # noqa: E402
import infer_v5 as I5  # noqa: E402
import torch_models_v4 as V4  # noqa: E402
from eval_kihmm_v4 import load_model  # noqa: E402
from train_kihmm_v4 import batch, load  # noqa: E402
from train_kihmm_v2 import group_batches  # noqa: E402

RESULTS = ROOT / "code" / "05_baselines" / "results"
MODEL = ROOT / "code" / "04_ml" / "models" / "kihmm_v4_v5a.pt"


def run_split(name: str, model, device: str) -> dict:
    data = load(*((name.split("_x") + ["1"])[:2] if "_x" in name else (name, "1")))
    n_groups = len(data["R"])
    n_head, rates_head, log_scale = [], [], []
    for gids in group_batches(n_groups, 8, np.random.default_rng(0), shuffle=False):
        b = batch(data, gids, "cpu")
        p = V4.predict(model, b["x"], b["group"], emission_chunk=250)
        n_head.append(p["N_hat"].cpu().numpy())
        rates_head.append(p["rates"].cpu().numpy())
        log_scale.append(p["log_scale"].cpu().numpy())
    n_head = np.concatenate(n_head)
    rates_head = np.concatenate(rates_head)
    log_scale = np.concatenate(log_scale)
    group = data["group"]
    model.emission.to(device)
    n_evid = I5.evidence_n(data["X"], rates_head[group], log_scale, model.emission,
                           device).argmax(axis=1) + 1
    model.emission.to("cpu")
    rates_ref, scale_ref = I5.refine(data["X"], n_evid, group, rates_head[group],
                                     log_scale, model.emission, "cpu", steps=25,
                                     prior=2.0, max_scale=2.5)
    counts, post_open = I5.posterior(data["X"], n_evid, rates_ref, scale_ref,
                                     model.emission, "cpu")
    return {"counts": counts, "N_pred": n_evid, "N_true": data["N"],
            "y": data["y"], "r": data["r"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="test_x1,test_x2,test_x4")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evidence_dev, _ = I5.pick_devices(args.device)
    device = evidence_dev if evidence_dev != "cpu" else "cpu"
    model = load_model(str(MODEL), "cpu")
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in [s.strip() for s in args.splits.split(",") if s.strip()]:
        t0 = time.time()
        out = run_split(name, model, device)
        np.savez_compressed(RESULTS / f"v5_{name}_predictions.npz", **out)
        open_acc = float((np.round(out["counts"][:, np.flatnonzero(B.STATEMAP == 1)].sum(axis=1))
                          == out["y"]).mean())
        n_acc = float((out["N_pred"] == out["N_true"]).mean())
        print(f"{name}: {len(out['N_true'])} traces, N acc {n_acc:.3f}, "
              f"open acc {open_acc:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

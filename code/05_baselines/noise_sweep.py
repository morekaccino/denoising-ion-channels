"""Noise-factor sweep for the baseline comparison graphs.

Evaluates every method at 10 equally spaced noise factors from 1.0 to 4.0 on
the frozen synth_v2 test set. The factors 1, 2 and 4 reuse the committed
results (they are the frozen x1/x2/x4 evaluations); the other seven levels are
synthesized from the frozen x1 traces with the generator's own noise formula
(see ``hmm_core.load_traces``), so all methods see identical traces per level.

Usage:
  python code/05_baselines/noise_sweep.py --method v5
  python code/05_baselines/noise_sweep.py --method sdmc --jobs 10
  python code/05_baselines/noise_sweep.py --method deepchannel
  python code/05_baselines/noise_sweep.py --method all --jobs 10   # sequential
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
RESULTS = ROOT / "code" / "05_baselines" / "results"

DEFAULT_LEVELS = [round(1.0 + i / 3.0, 6) for i in range(10)]
FROZEN = {1.0: "test_x1", 2.0: "test_x2", 4.0: "test_x4"}
BASELINE_TAGS = {"sdmc": "sdmc", "vnd": "vnd", "idc": "idc",
                 "deepchannel": "deepchannel"}


def metrics_from_open(path: np.ndarray, y: np.ndarray, N: np.ndarray,
                      N_hat: np.ndarray | None = None) -> dict:
    path = np.asarray(path)
    if N_hat is None:
        N_hat = path.max(axis=1)
    return {
        "n_acc": float((N_hat == N).mean()),
        "open_acc": float((path == y).mean()),
        "open_mae": float(np.abs(path.astype(np.float64) - y).mean()),
    }


def metrics_from_counts(counts: np.ndarray, y: np.ndarray, N: np.ndarray,
                        N_hat: np.ndarray) -> dict:
    open_expected = counts[:, (3, 4)].sum(axis=1)
    return {
        "n_acc": float((N_hat == N).mean()),
        "open_acc": float((np.round(open_expected) == y).mean()),
        "open_mae": float(np.abs(open_expected - y).mean()),
    }


def frozen_metrics(method: str, level: float) -> dict | None:
    if level not in FROZEN:
        return None
    split = FROZEN[level]
    if method == "v5":
        path = RESULTS / f"v5_{split}_predictions.npz"
        if not path.exists():
            return None
        with np.load(path) as d:
            return metrics_from_counts(d["counts"], d["y"], d["N_true"], d["N_pred"])
    tag = BASELINE_TAGS[method]
    path = RESULTS / f"{tag}_{split}.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text())["metrics"]
    return {"n_acc": m["n_acc"], "open_acc": m["open_acc"], "open_mae": m["open_mae"]}


# --------------------------------------------------------------------------- #
# per-method runners over an explicit X array
# --------------------------------------------------------------------------- #

def _hmm_worker(args):
    import importlib
    module, i, trace = args
    mod = importlib.import_module(module)
    lo, hi = np.percentile(trace, [0.5, 99.5])
    if module == "sdmc_port":
        out = mod.fit_model(trace, ls=(1, 2, 3, 4, 5), delta=0.01,
                            emit_range=(float(lo), float(hi)))
    else:
        out = mod.fit_model(trace, ls=(1, 2, 3, 4, 5),
                            emit_range=(float(lo), float(hi)))
    fit = out["fits"][out["best_l"]]
    return i, fit["path"].astype(np.int8), out["best_l"]


def run_hmm(method: str, X: np.ndarray, y: np.ndarray, N: np.ndarray,
            jobs: int) -> dict:
    module = f"{method}_port"
    paths = np.empty((len(X), X.shape[1]), dtype=np.int8)
    N_hat = np.empty(len(X), dtype=np.int64)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, path, L in ex.map(_hmm_worker, [(module, i, X[i]) for i in range(len(X))],
                                 chunksize=4):
            paths[i] = path
            N_hat[i] = L
    return metrics_from_open(paths, y, N, N_hat)


def _idc_worker(args):
    import idc_port
    i, trace = args
    out = idc_port.run_trace(trace)
    return i, out["path"].astype(np.int8), out["L_hat"]


def run_idc(X: np.ndarray, y: np.ndarray, N: np.ndarray, jobs: int) -> dict:
    paths = np.empty((len(X), X.shape[1]), dtype=np.int8)
    N_hat = np.empty(len(X), dtype=np.int64)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, path, L in ex.map(_idc_worker, [(i, X[i]) for i in range(len(X))],
                                 chunksize=4):
            paths[i] = path
            N_hat[i] = L
    return metrics_from_open(paths, y, N, N_hat)


def run_deepchannel(X: np.ndarray) -> dict:
    import torch
    import deepchannel_port as DC
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(DC.MODELS / "deepchannel.pt", map_location=device)
    lo, hi = ckpt["scale"]
    model = DC.DeepChannel().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    flat = ((X.reshape(-1) - lo) / (hi - lo)).astype(np.float32)
    pred = np.empty(len(flat), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(flat), 16384):
            xb = torch.from_numpy(flat[start:start + 16384]).float().to(device)[:, None]
            pred[start:start + 16384] = model(xb).argmax(dim=-1).cpu().numpy()
    paths = pred.reshape(X.shape)
    return paths


def v5_data(scale: float) -> dict:
    d = dict(np.load(core.SYNTH / "test.npz"))
    N = d["N"].astype(np.float64)
    y = d["y"].astype(np.float64)
    level = 0.58 * N[:, None] + 0.82 * y
    X = (level + (d["X_s1"].astype(np.float64) - level) * scale).astype(np.float32)
    K = int((d["group"] == 0).sum())
    return {"X": X, "y": d["y"], "r": d["r"], "N": d["N"], "group": d["group"],
            "R": d["R"], "log_scale": np.full(len(X), np.log(scale), dtype=np.float32),
            "K": K}


def run_v5(scale: float) -> dict:
    sys.path.insert(0, str(ROOT / "code" / "04_ml"))
    import infer_v5 as I5
    import torch_models_v4 as V4
    from eval_kihmm_v4 import load_model
    from train_kihmm_v2 import group_batches
    from train_kihmm_v4 import batch

    model_path = ROOT / "code" / "04_ml" / "models" / "kihmm_v4_v5a.pt"
    emission_dev, _ = I5.pick_devices("auto")
    model = load_model(str(model_path), "cpu")
    data = v5_data(scale)
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
    if emission_dev != "cpu":
        model.emission.to(emission_dev)
    n_evid = I5.evidence_n(data["X"], rates_head[group], log_scale, model.emission,
                           emission_dev).argmax(axis=1) + 1
    model.emission.to("cpu")
    rates_ref, scale_ref = I5.refine(data["X"], n_evid, group, rates_head[group],
                                     log_scale, model.emission, "cpu", steps=25,
                                     prior=2.0, max_scale=2.5)
    counts, _ = I5.posterior(data["X"], n_evid, rates_ref, scale_ref,
                             model.emission, "cpu")
    return metrics_from_counts(counts, data["y"], data["N"], n_evid)


RUNNERS = {
    "v5": lambda X, y, N, s, jobs: run_v5(s),
    "sdmc": lambda X, y, N, s, jobs: run_hmm("sdmc", X, y, N, jobs),
    "vnd": lambda X, y, N, s, jobs: run_hmm("vnd", X, y, N, jobs),
    "idc": lambda X, y, N, s, jobs: run_idc(X, y, N, jobs),
    "deepchannel": lambda X, y, N, s, jobs: metrics_from_open(
        run_deepchannel(X), y, N),
}


def sweep(method: str, levels: list[float], jobs: int) -> dict:
    data = core.load_traces("test")
    y, N = data["y"], data["N"]
    out = {"method": method, "traces": int(len(y)), "levels": levels,
           "n_acc": [], "open_acc": [], "open_mae": [], "reused": {}}
    synth_check = core.load_traces("test", scale=2.0)["X"]
    ref = core.load_traces("test_x2")["X"]
    out["synthesis_check_max_abs_diff"] = float(np.abs(synth_check - ref).max())
    for level in levels:
        cached = frozen_metrics(method, level)
        if cached is not None:
            out["reused"][str(level)] = True
            m = cached
        else:
            out["reused"][str(level)] = False
            t0 = time.time()
            X = core.load_traces("test", scale=level)["X"]
            m = RUNNERS[method](X, y, N, level, jobs)
            m["runtime_s"] = round(time.time() - t0, 1)
        out["n_acc"].append(m["n_acc"])
        out["open_acc"].append(m["open_acc"])
        out["open_mae"].append(m["open_mae"])
        print(f"{method} s={level:.4f}: N {m['n_acc']:.3f} open {m['open_acc']:.4f} "
              f"MAE {m['open_mae']:.4f} ({'cached' if cached else str(m.get('runtime_s')) + 's'})",
              flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"noise_sweep_{method}.json"
    path.write_text(json.dumps(out, indent=2))
    print("saved", path, flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="v5",
                        choices=["v5", "sdmc", "vnd", "idc", "deepchannel", "all"])
    parser.add_argument("--levels", default=",".join(str(v) for v in DEFAULT_LEVELS))
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()
    levels = [float(v) for v in args.levels.split(",") if v.strip()]
    methods = list(RUNNERS) if args.method == "all" else [args.method]
    for method in methods:
        sweep(method, levels, args.jobs)


if __name__ == "__main__":
    main()

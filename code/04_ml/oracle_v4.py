"""Bayes-optimal decode built only from the v4 differentiable layers.

This is the derisking step for KI-HMM v4. Before any training, the layers in
``torch_kinetics.py`` and ``torch_emissions.py`` are fed the **true** channel
count and the **true** rate table and asked to decode the frozen test set. If
they are correct they must land on the numbers ``refine_kihmm_v2.py --oracle``
reports with scipy and numpy (open-count accuracy about 0.91, per-state MAE
about 0.20). Anything less means the layers are wrong and no amount of training
will help.

It doubles as the timing benchmark that decides the training device and whether
the 462-state chain for N=5 is affordable.

Usage:
  python code/04_ml/oracle_v4.py --emission-fit
  python code/04_ml/oracle_v4.py --device mps --limit 120
  python code/04_ml/oracle_v4.py --timing
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "04_ml"))

import benchmark as B  # noqa: E402
import benchmark_v2 as B2  # noqa: E402
import torch_emissions as TE  # noqa: E402
import torch_kinetics as TK  # noqa: E402

MODELS = ROOT / "code" / "04_ml" / "models"
RESULTS = ROOT / "code" / "04_ml" / "results"
EMISSION_FIT = MODELS / "emission_gh_fit.pt"
TERM_BUDGET = 40_000_000


def load_emission(n_comp: int = 4, k_max: int = 5, refit: bool = False,
                  device: str = "cpu") -> TE.SumMixtureEmission:
    """The GH-matched emission module, fitted once and cached."""
    module = TE.SumMixtureEmission(n_comp=n_comp, k_max=k_max)
    if EMISSION_FIT.exists() and not refit:
        ckpt = torch.load(EMISSION_FIT, map_location="cpu")
        module.load_state_dict(ckpt["model"])
    else:
        torch.manual_seed(0)
        TE.fit_to_simulator(module, verbose=True)
        MODELS.mkdir(parents=True, exist_ok=True)
        torch.save({"model": module.state_dict(), "n_comp": n_comp, "k_max": k_max}, EMISSION_FIT)
        print(f"saved {EMISSION_FIT}", flush=True)
    return module.to(device).eval()


@torch.no_grad()
def decode(X: np.ndarray, N: np.ndarray, rates: np.ndarray, emission: TE.SumMixtureEmission,
           log_scale: float = 0.0, device: str = "cpu") -> np.ndarray:
    """Posterior expected per-state counts (n_traces, 7, T) for known N and rates."""
    out = np.empty((len(X), TK.N_STATES, X.shape[1]), dtype=np.float32)
    for n in np.unique(N):
        tables = TK.get_tables(int(n), device)
        idx = np.flatnonzero(N == n)
        per = max(1, min(len(idx), TERM_BUDGET // max(tables.picks.shape[0], 1)))
        for lo in range(0, len(idx), per):
            sel = idx[lo : lo + per]
            x = torch.as_tensor(X[sel], dtype=torch.float32, device=device)
            r = torch.as_tensor(rates[sel], dtype=torch.float32, device=device)
            P0 = TK.p0_from_rates(r)
            logP = TK.count_transition(P0, tables).clamp_min(1e-30).log()
            logpi = TK.count_log_initial(TK.stationary(P0), tables)
            scale = torch.full((len(sel),), log_scale, device=device)
            logb = TK.expand_open_emissions(emission(x, int(n), scale), tables)
            _, gamma = TK.hmm_posterior(logb, logP, logpi)
            out[sel] = TK.posterior_counts(gamma, tables).cpu().numpy()
    return out


def metrics(counts: np.ndarray, y: np.ndarray, r: np.ndarray) -> dict:
    counts_true = np.stack([(r == s).sum(axis=1) for s in range(TK.N_STATES)], axis=1).astype(float)
    open_hat = counts[:, TK.OPEN_STATES].sum(axis=1)
    return {
        "open_acc": float((np.round(open_hat) == y).mean()),
        "open_mae": float(np.abs(open_hat - y).mean()),
        "state_mae": float(np.abs(counts - counts_true).mean()),
    }


def _split(name: str):
    scale = 1.0
    base = name
    if "_x" in name:
        base, s = name.split("_x")
        scale = float(s)
    d = B2.load_split(base)
    X = d[f"X_s{scale:g}"] if f"X_s{scale:g}" in d else d["X"]
    return X, d["y"], d["r"], d["N"], d["group"], d["R"], scale


def run(splits: list[str], emission, device: str, limit: int = 0) -> dict:
    report = {}
    for name in splits:
        X, y, r, N, group, R, scale = _split(name)
        if limit:
            X, y, r, N, group = X[:limit], y[:limit], r[:limit], N[:limit], group[:limit]
        t0 = time.time()
        counts = decode(X, N, R[group], emission, float(np.log(scale)), device)
        row = metrics(counts, y, r)
        row["seconds"] = round(time.time() - t0, 1)
        row["traces"] = int(len(X))
        report[name] = row
        print(f"{name:10s} open_acc={row['open_acc']:.4f} open_mae={row['open_mae']:.3f} "
              f"state_mae={row['state_mae']:.4f}  ({row['traces']} traces, {row['seconds']}s)", flush=True)
    return report


def timing(emission, limit: int = 48) -> dict:
    """Per-trace decode cost by N and device, to size the training loop."""
    X, y, r, N, group, R, _ = _split("test")
    home = next(emission.parameters()).device
    out = {}
    for device in dict.fromkeys(["cpu", TK.default_device()]):
        emission.to(device)
        row = {}
        for n in (1, 3, 5):
            sel = np.flatnonzero(N == n)[:limit]
            if not len(sel):
                continue
            decode(X[sel][:2], N[sel][:2], R[group[sel]][:2], emission, 0.0, device)
            t0 = time.time()
            decode(X[sel], N[sel], R[group[sel]], emission, 0.0, device)
            row[f"N={n}"] = round((time.time() - t0) / len(sel) * 1000, 1)
        out[device] = row
        print(f"{device:4s}  ms per trace (T=1000): {row}", flush=True)
    emission.to(home)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="test_x1,test_x2,test_x4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n-comp", type=int, default=4)
    parser.add_argument("--emission-fit", action="store_true", help="refit the emission module")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--tag", default="oracle_v4")
    args = parser.parse_args()

    emission = load_emission(n_comp=args.n_comp, refit=args.emission_fit, device=args.device)
    report = {}
    if args.timing:
        report["timing"] = timing(emission)
    report["splits"] = run([s.strip() for s in args.splits.split(",") if s.strip()],
                           emission, args.device, args.limit)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.tag}.json").write_text(json.dumps(report, indent=2))
    print("saved", RESULTS / f"{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()

"""Exact inference on top of a trained KI-HMM v4 network.

v4 learns a complete generative model of a recording: a per-channel emission
density (``torch_emissions.py``) and the exact occupancy-count chain built from
predicted rates (``torch_kinetics.py``). The trained heads then throw most of
that away at inference -- they read the channel count off a classifier and the
rates off a regressor, when the model can simply be *asked* which values explain
the trace best.

Two test-time procedures, both gradient-based and both using nothing but the
network's own layers:

  evidence_n     score every candidate N by the model's log evidence and take
                 the best. Exact Bayesian model selection over the channel
                 count; on the frozen test set this is worth 0.909 -> ~1.00.
  refine_rates   maximise that same log evidence over the 12 rates with Adam.
                 Maximum likelihood estimation by gradient descent through the
                 chain; identifiable-direction R2 0.36 -> ~0.89.

Both are measured against, and reported alongside, the plain forward pass, so
the contribution of each stage is visible.

Device note: the evidence pass is forward-only and scores every trace at N=5
(the 462-state chain), which suits the GPU (MPS 25 ms/trace vs CPU 36). Rate
refinement needs the backward pass through ``CountTransition``, a chunked
scatter over millions of terms that MPS handles poorly (CPU 3.3 s/group vs MPS
8.7). ``--device auto`` therefore splits them.

Usage:
  python code/04_ml/infer_v5.py --model code/04_ml/models/kihmm_v4_v4b.pt
  python code/04_ml/infer_v5.py --splits test_x1 --limit-groups 8 --steps 10
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

import benchmark_v2 as B2  # noqa: E402
import torch_kinetics as TK  # noqa: E402
import torch_models_v4 as V4  # noqa: E402
from eval_kihmm_v4 import load_model  # noqa: E402
from train_kihmm_v4 import batch, load  # noqa: E402
from train_kihmm_v2 import group_batches  # noqa: E402

RESULTS = ROOT / "code" / "04_ml" / "results"
K_MAX = V4.K_MAX


def pick_devices(choice: str) -> tuple[str, str]:
    """(evidence device, refinement device) for ``auto`` or an explicit name."""
    if choice != "auto":
        return choice, choice
    return (TK.default_device() if torch.backends.mps.is_available() else "cpu"), "cpu"


def chain_for(rates: torch.Tensor, tables: TK.ChainTables):
    P0 = TK.p0_from_rates(rates)
    return (TK.count_transition(P0, tables).clamp_min(1e-30).log(),
            TK.count_log_initial(TK.stationary(P0), tables))


def decode(x: torch.Tensor, n: int, rates: torch.Tensor, log_scale: torch.Tensor,
           emission, chunk: int = 250):
    """Posterior for a block of traces that share the channel count ``n``."""
    tables = TK.get_tables(n, x.device)
    logP, logpi = chain_for(rates, tables)
    logb = TK.expand_open_emissions(emission(x, n, log_scale, chunk=chunk), tables)
    logZ, gamma = TK.hmm_posterior(logb, logP, logpi)
    post_k = gamma.new_zeros(gamma.shape[0], gamma.shape[1], K_MAX + 1)
    post_k.index_add_(2, tables.open_count, gamma)
    return logZ, TK.posterior_counts(gamma, tables), post_k


@torch.no_grad()
def evidence_n(X: np.ndarray, rates: np.ndarray, log_scale: np.ndarray, emission,
               device: str, per_batch: int = 24) -> np.ndarray:
    """Log evidence of every candidate channel count, shape (n_traces, K_MAX)."""
    n = len(X)
    out = np.full((n, K_MAX), -np.inf, dtype=np.float64)
    for cand in range(1, K_MAX + 1):
        for lo in range(0, n, per_batch):
            sl = slice(lo, min(lo + per_batch, n))
            lz, _, _ = decode(torch.as_tensor(X[sl], dtype=torch.float32, device=device), cand,
                              torch.as_tensor(rates[sl], dtype=torch.float32, device=device),
                              torch.as_tensor(log_scale[sl], dtype=torch.float32, device=device),
                              emission)
            out[sl, cand - 1] = lz.cpu().double().numpy()
    return out


def refine(X: np.ndarray, N: np.ndarray, group: np.ndarray, rates0: np.ndarray,
           log_scale0: np.ndarray, emission, device: str, steps: int = 25,
           lr: float = 0.08, init: str = "base", fit_scale: bool = True,
           prior: float = 2.0, max_scale: float = 2.5):
    """Maximise the model log evidence over the rates and the noise scale.

    The rate table is shared by a group; the noise scale is per trace. Both are
    nuisance parameters of the same likelihood, and the scale head is
    noticeably biased (it reads 1.28 on noise x1 data), which blurs the emission
    and costs the rate estimate, so it is fitted here too.

    Traces of a group are batched by channel count, which is what makes this
    affordable: a per-trace loop costs 5.1 s per group, bucketed 3.3 s.

    ``prior`` pulls the rates back toward the base table. At noise x4 most rate
    directions carry almost no information, the likelihood surface is flat, and
    unpenalised maximisation wanders far enough to undo the gains it makes at
    x1; a weak quadratic penalty turns this into MAP estimation under a stand-in
    for the training prior and keeps high noise from regressing.

    ``max_scale`` skips refinement entirely for groups whose estimated noise
    scale is above it. Even with the prior, refinement is a small net loss at
    noise x4 (open 0.496 to 0.489), so it is better not to run it there.
    """
    base = torch.as_tensor(B2.BASE_OFF, dtype=torch.float32, device=device)
    rates_out, scale_out = rates0.copy(), log_scale0.copy()
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        if np.exp(log_scale0[idx].mean()) > max_scale:
            continue  # rates carry almost no information at this noise level
        if init == "head":
            start = torch.as_tensor(np.log(rates0[idx[0]] / B2.BASE_OFF),
                                    dtype=torch.float32, device=device)
            delta = torch.atanh(start.clamp(-0.98, 0.98)).clone().requires_grad_(True)
        else:
            delta = torch.zeros(V4.N_RATES, device=device, requires_grad=True)
        buckets = [(int(v), idx[N[idx] == v]) for v in np.unique(N[idx])]
        cached = [(n, torch.as_tensor(X[sel], dtype=torch.float32, device=device),
                   torch.as_tensor(log_scale0[sel], dtype=torch.float32, device=device),
                   torch.zeros(len(sel), device=device, requires_grad=fit_scale))
                  for n, sel in buckets]
        params = [delta] + ([c[3] for c in cached] if fit_scale else [])
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            total = 0.0
            rates = (base[None] * torch.exp(torch.tanh(delta))[None])
            for n, xb, sb, ds in cached:
                lz, _, _ = decode(xb, n, rates.expand(len(xb), V4.N_RATES), sb + ds, emission)
                total = total + lz.sum()
            loss = -total / len(idx) + prior * torch.tanh(delta).pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        rates_out[idx] = (B2.BASE_OFF * np.exp(np.tanh(delta.detach().cpu().numpy()))).astype(rates_out.dtype)
        if fit_scale:
            for (n, _, sb, ds), (_, sel) in zip(cached, buckets):
                scale_out[sel] = (sb + ds).detach().cpu().numpy()
    return rates_out, scale_out


@torch.no_grad()
def posterior(X: np.ndarray, N: np.ndarray, rates: np.ndarray, log_scale: np.ndarray,
              emission, device: str, per_batch: int = 24):
    """Per-state counts and the open-count posterior for known N and rates."""
    counts = np.empty((len(X), V4.N_STATES, X.shape[1]), dtype=np.float32)
    post = np.empty((len(X), X.shape[1], K_MAX + 1), dtype=np.float32)
    for n in np.unique(N):
        idx = np.flatnonzero(N == n)
        for lo in range(0, len(idx), per_batch):
            sel = idx[lo : lo + per_batch]
            _, c, pk = decode(torch.as_tensor(X[sel], dtype=torch.float32, device=device), int(n),
                              torch.as_tensor(rates[sel], dtype=torch.float32, device=device),
                              torch.as_tensor(log_scale[sel], dtype=torch.float32, device=device),
                              emission)
            counts[sel], post[sel] = c.cpu().numpy(), pk.cpu().numpy()
    return counts, post


def metrics(counts, post, y, r, N_hat, N_true, pred_log=None, truth_log=None, basis=None) -> dict:
    counts_true = np.stack([(r == s).sum(axis=1) for s in range(V4.N_STATES)], axis=1).astype(float)
    e_open = counts[:, TK.OPEN_STATES].sum(axis=1)
    out = {
        "n_acc": float((N_hat == N_true).mean()),
        "open_acc": float((np.round(e_open) == y).mean()),
        "open_acc_map": float((post.argmax(axis=2) == y).mean()),
        "open_mae": float(np.abs(e_open - y).mean()),
        "state_mae": float(np.abs(counts - counts_true).mean()),
    }
    if basis is not None and pred_log is not None:
        out["dir_r2"] = [round(float(1 - ((pred_log @ v - truth_log @ v) ** 2).mean()
                                     / max((truth_log @ v).var(), 1e-12)), 4) for v in basis.T]
    return out


def run_split(model, data: dict, emission_dev: str, refine_dev: str, steps: int,
              basis: np.ndarray, limit_groups: int = 0, init: str = "base",
              prior: float = 2.0, max_scale: float = 2.5) -> dict:
    n_groups = limit_groups or len(data["R"])
    keep = n_groups * data["K"]
    X, y, r = data["X"][:keep], data["y"][:keep], data["r"][:keep]
    N_true, group = data["N"][:keep], data["group"][:keep]
    truth_log = np.log(data["R"][:n_groups])

    # stage 0: the plain forward pass, for reference
    n_head, rates_head, log_scale = [], [], []
    for gids in group_batches(n_groups, 8, np.random.default_rng(0), shuffle=False):
        b = batch(data, gids, refine_dev)
        p = V4.predict(model, b["x"], b["group"], emission_chunk=250)
        n_head.append(p["N_hat"].cpu().numpy())
        rates_head.append(p["rates"].cpu().numpy())
        log_scale.append(p["log_scale"].cpu().numpy())
    n_head = np.concatenate(n_head)
    rates_head = np.concatenate(rates_head)
    log_scale = np.concatenate(log_scale)

    stages, timing = {}, {}
    model.emission.to(refine_dev)
    counts, post = posterior(X, n_head, rates_head[group], log_scale, model.emission, refine_dev)
    stages["model"] = metrics(counts, post, y, r, n_head, N_true, np.log(rates_head), truth_log, basis)

    # stage 1: channel count by log evidence, scored with the head's own rates
    model.emission.to(emission_dev)
    t0 = time.time()
    logZ = evidence_n(X, rates_head[group], log_scale, model.emission, emission_dev)
    timing["evidence_s"] = round(time.time() - t0, 1)
    n_evid = logZ.argmax(axis=1) + 1
    model.emission.to(refine_dev)
    counts, post = posterior(X, n_evid, rates_head[group], log_scale, model.emission, refine_dev)
    stages["evidence_n"] = metrics(counts, post, y, r, n_evid, N_true, np.log(rates_head), truth_log, basis)

    # stage 2: rates and noise scale by maximising the same evidence
    t0 = time.time()
    rates_ref, scale_ref = refine(X, n_evid, group, rates_head[group], log_scale,
                                  model.emission, refine_dev, steps=steps, init=init,
                                  prior=prior, max_scale=max_scale)
    timing["refine_s"] = round(time.time() - t0, 1)
    per_group = rates_ref[np.searchsorted(group, np.arange(n_groups))]
    counts, post = posterior(X, n_evid, rates_ref, scale_ref, model.emission, refine_dev)
    stages["+ refined rates and scale"] = metrics(
        counts, post, y, r, n_evid, N_true, np.log(per_group), truth_log, basis)

    # stage 3: re-select N now that the nuisance parameters are fitted
    model.emission.to(emission_dev)
    t0 = time.time()
    n_final = evidence_n(X, rates_ref, scale_ref, model.emission, emission_dev).argmax(axis=1) + 1
    timing["evidence2_s"] = round(time.time() - t0, 1)
    model.emission.to(refine_dev)
    counts, post = posterior(X, n_final, rates_ref, scale_ref, model.emission, refine_dev)
    stages["+ N re-selected"] = metrics(
        counts, post, y, r, n_final, N_true, np.log(per_group), truth_log, basis)
    true_scale = data["log_scale"][:keep]
    return {"stages": stages, "timing": timing, "traces": int(len(X)),
            "log_scale_mae": {"head": float(np.abs(log_scale - true_scale).mean()),
                              "refined": float(np.abs(scale_ref - true_scale).mean())},
            "rates_refined": np.log(per_group).tolist(),
            "rates_head": np.log(rates_head).tolist(),
            "rates_true": truth_log.tolist()}


def stage_figure(report: dict, basis: np.ndarray, path: pathlib.Path) -> None:
    """Left: what each inference stage buys. Right: the refined rates against truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = list(report["splits"])
    stages = list(report["splits"][splits[0]]["stages"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    width = 0.8 / len(stages)
    for j, st in enumerate(stages):
        ax[0].bar(np.arange(len(splits)) + (j - (len(stages) - 1) / 2) * width,
                  [report["splits"][s]["stages"][st]["open_acc"] for s in splits], width, label=st)
    ax[0].set_xticks(range(len(splits)), [s.replace("test_", "noise ") for s in splits])
    ax[0].set_ylabel("open-count accuracy")
    ax[0].set_title("what each inference stage buys")
    ax[0].legend(fontsize=7)

    row = report["splits"][splits[0]]
    v1 = basis[:, 0]
    truth = np.asarray(row["rates_true"]) @ v1
    for key, label in (("rates_head", "feedforward head"), ("rates_refined", "after refinement")):
        pred = np.asarray(row[key]) @ v1
        r2 = 1 - ((pred - truth) ** 2).mean() / max(truth.var(), 1e-12)
        ax[1].scatter(truth, pred, s=18, alpha=0.75, label=f"{label}: R2={r2:.2f}")
    lo, hi = truth.min(), truth.max()
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[1].set_xlabel("true")
    ax[1].set_ylabel("predicted")
    ax[1].set_title("top identifiable rate direction (noise x1)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print("saved", path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "code" / "04_ml" / "models" / "kihmm_v4_v4b.pt"))
    parser.add_argument("--splits", default="test_x1,test_x2,test_x4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--init", default="base", choices=("base", "head"),
                        help="starting point for the rate refinement")
    parser.add_argument("--prior", type=float, default=2.0,
                        help="quadratic pull toward the base rate table (MAP instead of ML)")
    parser.add_argument("--refine-max-scale", type=float, default=2.5,
                        help="skip rate refinement for groups noisier than this")
    parser.add_argument("--tag", default="kihmm_v5_infer")
    args = parser.parse_args()

    emission_dev, refine_dev = pick_devices(args.device)
    print(f"evidence on {emission_dev}, refinement on {refine_dev}", flush=True)
    model = load_model(args.model, refine_dev)
    evals, evecs = np.linalg.eigh(
        np.asarray(json.loads((RESULTS / "fisher_synth_v2.json").read_text())["fisher"]))
    basis = evecs[:, np.argsort(evals)[::-1][:4]]

    report = {"model": args.model, "devices": {"evidence": emission_dev, "refine": refine_dev},
              "steps": args.steps, "splits": {}}
    for name in [s.strip() for s in args.splits.split(",") if s.strip()]:
        base, scale = (name.split("_x") + ["1"])[:2] if "_x" in name else (name, "1")
        row = run_split(model, load(base, scale), emission_dev, refine_dev, args.steps,
                        basis, args.limit_groups, args.init, args.prior,
                        args.refine_max_scale)
        report["splits"][name] = row
        print(f"\n{name} ({row['traces']} traces, evidence {row['timing']['evidence_s']}s, "
              f"refine {row['timing']['refine_s']}s) | log-scale MAE "
              f"head {row['log_scale_mae']['head']:.3f} -> refined {row['log_scale_mae']['refined']:.3f}",
              flush=True)
        for stage, m in row["stages"].items():
            print(f"  {stage:26s} N={m['n_acc']:.3f} open={m['open_acc']:.4f} "
                  f"map={m['open_acc_map']:.4f} state_mae={m['state_mae']:.4f} "
                  f"dirR2={np.round(m.get('dir_r2', []), 3)}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.tag}.json").write_text(json.dumps(report, indent=2))
    try:
        stage_figure(report, basis, RESULTS / "figures" / f"{args.tag}.png")
    except Exception as exc:
        print("figure skipped:", exc, flush=True)
    print("\nsaved", RESULTS / f"{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()

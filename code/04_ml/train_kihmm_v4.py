"""Train KI-HMM v4 on the rate-randomized benchmark (synth_v2).

Training is plain gradient descent on one network. The structured part of the
loss is the CRF identity log p(states | y) = path_score - logZ, which needs no
backpropagation through the forward-backward recursion, plus a generative -logZ
term that holds the learned emission density and the predicted rates to the
data.

Traces are cropped during training (the chain is stationary, so any window is a
valid sample) and the forward-backward runs bucketed by channel count so an
N=1 trace never pays for the 462-state chain of an N=5 trace.

Artifacts:
  code/04_ml/models/kihmm_v4_<tag>.pt
  code/04_ml/results/kihmm_v4_<tag>.json

Usage:
  python code/04_ml/train_kihmm_v4.py --epochs 60 --tag v4a
  python code/04_ml/train_kihmm_v4.py --epochs 2 --limit-groups 16 --tag smoke
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
from train_kihmm_v2 import compute_fisher, group_batches  # noqa: E402

MODELS = ROOT / "code" / "04_ml" / "models"
RESULTS = ROOT / "code" / "04_ml" / "results"
EMISSION_FIT = MODELS / "emission_gh_fit.pt"


def load(name: str, scale: str | None = None) -> dict:
    d = B2.load_split(name)
    X = d[f"X_s{scale}"] if scale else (d["X"] if "X" in d else d["X_s1"])
    log_scale = (np.log(d["trace_scale"]).astype(np.float32) if "trace_scale" in d
                 else np.full(len(X), np.log(float(scale or 1.0)), dtype=np.float32))
    return {"X": X, "y": d["y"], "r": d["r"], "N": d["N"], "group": d["group"],
            "R": d["R"], "log_scale": log_scale, "K": int((d["group"] == 0).sum())}


def load_concat(names: str) -> dict:
    parts = [load(n.strip()) for n in names.split(",") if n.strip()]
    if len(parts) == 1:
        return parts[0]
    if len({p["K"] for p in parts}) != 1:
        raise ValueError("splits must share the same traces-per-group")
    out, offset = {"K": parts[0]["K"]}, 0
    for key in ("X", "y", "r", "N", "log_scale"):
        out[key] = np.concatenate([p[key] for p in parts])
    out["R"] = np.concatenate([p["R"] for p in parts])
    groups = []
    for p in parts:
        groups.append(p["group"] + offset)
        offset += len(p["R"])
    out["group"] = np.concatenate(groups)
    return out


def batch(data: dict, gids: np.ndarray, device: str, crop: int = 0,
          rng: np.random.Generator | None = None, k_use: int = 0):
    K = data["K"]
    idx = (np.asarray(gids)[:, None] * K + np.arange(k_use or K)[None, :]).reshape(-1)
    T_full = data["X"].shape[1]
    sl = slice(0, T_full)
    if crop and crop < T_full:
        start = int((rng or np.random.default_rng()).integers(0, T_full - crop + 1))
        sl = slice(start, start + crop)
    tt = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    return {
        "x": tt(data["X"][idx][:, sl], torch.float32),
        "r": tt(data["r"][idx][:, :, sl].astype(np.int64), torch.long),
        "y": tt(data["y"][idx][:, sl].astype(np.int64), torch.long),
        "N": tt(data["N"][idx].astype(np.int64), torch.long),
        "R": tt(data["R"][gids], torch.float32),
        "log_scale": tt(data["log_scale"][idx], torch.float32),
        "group": torch.arange(len(gids), device=device).repeat_interleave(k_use or K),
        "idx": idx,
    }


@torch.no_grad()
def evaluate(model: V4.KIHMMv4, data: dict, device: str, per_batch: int = 8,
             basis: np.ndarray | None = None, limit_groups: int = 0, k_use: int = 0) -> dict:
    model.eval()
    n_groups = limit_groups or len(data["R"])
    n_ok = open_ok = open_total = n_traces = 0
    state_abs = open_abs = 0.0
    preds, truths = [], []
    for gids in group_batches(n_groups, per_batch, np.random.default_rng(0), shuffle=False):
        b = batch(data, gids, device, k_use=k_use)
        p = V4.predict(model, b["x"], b["group"], emission_chunk=250)
        counts_true = V4.counts_from_r(b["r"])
        n_ok += int((p["N_hat"] == b["N"]).sum())
        n_traces += len(b["N"])
        state_abs += float((p["counts"] - counts_true).abs().sum())
        open_ok += int((torch.round(p["e_open"]) == b["y"]).sum())
        open_abs += float((p["e_open"] - b["y"]).abs().sum())
        open_total += b["y"].numel()
        preds.append(np.log(p["rates"].cpu().numpy()))
        truths.append(np.log(b["R"].cpu().numpy()))
    pred, truth = np.concatenate(preds), np.concatenate(truths)
    out = {
        "n_acc": n_ok / max(n_traces, 1),
        "state_mae": state_abs / max(open_total * V4.N_STATES, 1),
        "open_acc": open_ok / max(open_total, 1),
        "open_mae": open_abs / max(open_total, 1),
        "rate_mae": float(np.abs(pred - truth).mean()),
        "rate_r2_per_rate": [round(float(1 - ((pred[:, j] - truth[:, j]) ** 2).mean()
                                         / max(truth[:, j].var(), 1e-12)), 4)
                             for j in range(truth.shape[1])],
    }
    if basis is not None:
        r2 = [float(1 - ((pred @ v - truth @ v) ** 2).mean() / max((truth @ v).var(), 1e-12))
              for v in basis.T]
        out["dir_r2"] = [round(v, 4) for v in r2]
        out["dir_r2_mean"] = float(np.mean(r2))
    return out


def fisher_basis(device: str):
    path = RESULTS / "fisher_synth_v2.json"
    if path.exists():
        F = np.asarray(json.loads(path.read_text())["fisher"])
    else:
        F = compute_fisher(B2.load_split("val"))
        RESULTS.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fisher": F.tolist()}, indent=2))
    evals, evecs = np.linalg.eigh(F)
    keep = evals > 0.01 * evals.max()
    W = (evecs[:, keep] * evals[keep]) @ evecs[:, keep].T
    W = W / np.trace(W) * len(B2.RATE_INDEX)
    return (torch.as_tensor(W, dtype=torch.float32, device=device),
            evecs[:, np.argsort(evals)[::-1][:4]], int(keep.sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-groups", type=int, default=16)
    parser.add_argument("--crop", type=int, default=250)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--head-lr-mult", type=float, default=2.0,
                        help="learning-rate multiplier for the trace-level heads")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-comp", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--tag", default="v4a")
    parser.add_argument("--train-splits", default="train")
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--init-emission", action="store_true",
                        help="warm-start the emission mixture from the GH fit")
    parser.add_argument("--rate-stats", action="store_true",
                        help="give the rate head autocorrelation and transition statistics")
    parser.add_argument("--n-head", default="level", choices=("level", "trace", "pooled"),
                        help="what the channel-count head reads")
    parser.add_argument("--w-n", type=float, default=0.5)
    parser.add_argument("--w-crf", type=float, default=1.0)
    parser.add_argument("--w-gen", type=float, default=0.3)
    parser.add_argument("--w-rate", type=float, default=1.0)
    parser.add_argument("--w-scale", type=float, default=0.1)
    args = parser.parse_args()

    device = args.device or TK.default_device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train = load_concat(args.train_splits)
    val = load("val")
    n_groups = args.limit_groups or len(train["R"])
    rate_metric, basis, n_keep = fisher_basis(device)
    print(f"device={device} train={n_groups} groups x {train['K']} traces "
          f"crop={args.crop} fisher keeps {n_keep}/12", flush=True)

    model = V4.KIHMMv4(hidden=args.hidden, n_comp=args.n_comp, rate_stats=args.rate_stats,
                       n_head=args.n_head).to(device)
    if args.init_emission and EMISSION_FIT.exists():
        model.load_emission_fit(EMISSION_FIT)
        model.to(device)
        print(f"emission warm-started from {EMISSION_FIT.name}", flush=True)

    # The trace-level heads are throttled when they share a global gradient clip
    # with the structured losses: trained on their own, the count head reaches
    # 0.99 on the frozen test set but only 0.91 inside the joint model, and that
    # holds even when it shares no parameters with the encoder. Clipping the
    # heads separately, with their own learning rate, removes the coupling.
    head_modules = [model.n_head, model.scale_head, model.rate_dense, model.rate_head]
    head_params = [p for m in head_modules for p in m.parameters()]
    head_ids = {id(p) for p in head_params}
    core_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.Adam([{"params": core_params, "lr": args.lr},
                            {"params": head_params, "lr": args.lr * args.head_lr_mult}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, history = -np.inf, []
    val_groups = min(len(val["R"]), 32)  # cheap per-epoch view; full splits at the end
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        losses, parts = [], []
        for gids in group_batches(n_groups, args.batch_groups, rng):
            b = batch(train, gids, device, crop=args.crop, rng=rng)
            out = model(b["x"], b["group"], b["N"], emission_chunk=0)
            loss, info = V4.kihmm_v4_loss(
                out, b["N"], V4.counts_from_r(b["r"]), b["R"][b["group"]], b["log_scale"],
                w_n=args.w_n, w_crf=args.w_crf, w_gen=args.w_gen, w_rate=args.w_rate,
                w_scale=args.w_scale, rate_metric=rate_metric)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core_params, 1.0)
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            parts.append(info)
        sched.step()

        vm = evaluate(model, val, device, basis=basis, limit_groups=val_groups)
        score = vm["open_acc"] - 0.3 * vm["state_mae"] + 0.3 * vm["dir_r2_mean"] + 0.2 * vm["n_acc"]
        mean_parts = {k: float(np.mean([p[k] for p in parts])) for k in parts[0]}
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **mean_parts,
               **{f"val_{k}": v for k, v in vm.items()}, "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(f"[{epoch:02d}] loss={row['loss']:.3f} crf={mean_parts['crf']:.3f} "
              f"gen={mean_parts['gen']:.3f} | N={vm['n_acc']:.3f} open={vm['open_acc']:.4f} "
              f"state_mae={vm['state_mae']:.3f} dirR2={np.round(vm['dir_r2'], 3)} "
              f"({row['seconds']}s)", flush=True)
        if score > best:
            best = score
            MODELS.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                       MODELS / f"kihmm_v4_{args.tag}.pt")

    model.load_state_dict(torch.load(MODELS / f"kihmm_v4_{args.tag}.pt", map_location=device)["model"])
    full = 0 if not args.limit_groups else val_groups
    report = {"history": history, "final": {"val": evaluate(model, val, device, basis=basis,
                                                            limit_groups=full)},
              "test": {}, "extrap": {}}
    for s in ["1", "2", "4"]:
        report["test"][f"scale_{s}"] = evaluate(model, load("test", s), device, basis=basis,
                                                limit_groups=full)
        print(f"test x{s}: { {k: round(v, 4) for k, v in report['test'][f'scale_{s}'].items() if isinstance(v, float)} }", flush=True)
    for s in ["1", "2"]:
        report["extrap"][f"scale_{s}"] = evaluate(model, load("extrap", s), device, basis=basis,
                                                  limit_groups=full)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"kihmm_v4_{args.tag}.json").write_text(json.dumps(report, indent=2))
    print("saved", MODELS / f"kihmm_v4_{args.tag}.pt", flush=True)


if __name__ == "__main__":
    main()

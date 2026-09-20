"""Train KI-HMM v3 (neural HMM head) on the rate-randomized benchmark.

Artifacts:
  code/04_ml/models/kihmm_v3_<tag>.pt
  code/04_ml/results/kihmm_v3_<tag>.json

Usage:
  python code/04_ml/train_kihmm_v3.py --epochs 100 --batch-groups 24 --tag v3a
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
import torch_models_v2 as T2  # noqa: E402
import torch_models_v3 as V3  # noqa: E402
from train_kihmm_v2 import batch_tensors, compute_fisher, group_batches, load, load_concat  # noqa: E402

MODELS = ROOT / "code" / "04_ml" / "models"
RESULTS = ROOT / "code" / "04_ml" / "results"


@torch.no_grad()
def evaluate_v3(model, data, per_batch: int = 16, basis: np.ndarray | None = None) -> dict:
    X, y, r, N, group, R, K = data
    model.eval()
    counts_all, e_open_all, n_hat_all, y_all, r_all = [], [], [], [], []
    rate_preds, rate_truths = [], []
    for gids in group_batches(len(R), per_batch, np.random.default_rng(0), shuffle=False):
        idx = (np.asarray(gids)[:, None] * K + np.arange(K)[None, :]).reshape(-1)
        xb, rb, Nb, Rb, gb = batch_tensors(X, r, N, R, K, gids)
        p = V3.predict_v3(model, xb, gb)
        counts_all.append(p["counts"].cpu().numpy())
        e_open_all.append(p["e_open"].cpu().numpy())
        n_hat_all.append(p["N_hat"].cpu().numpy())
        rate_preds.append(np.log(p["rates"].cpu().numpy()))
        rate_truths.append(np.log(Rb.cpu().numpy()))
        y_all.append(y[idx])
        r_all.append(r[idx])
    counts = np.concatenate(counts_all)
    e_open = np.concatenate(e_open_all)
    n_hat = np.concatenate(n_hat_all)
    y = np.concatenate(y_all)
    r = np.concatenate(r_all)
    pred, truth = np.concatenate(rate_preds), np.concatenate(rate_truths)
    counts_true = np.stack([(r == s).sum(axis=1) for s in range(7)], axis=1).astype(float)
    out = {
        "n_acc": float((n_hat == N).mean()),
        "state_mae": float(np.abs(counts - counts_true).mean()),
        "open_acc": float((np.round(e_open) == y).mean()),
        "open_mae": float(np.abs(e_open - y).mean()),
    }
    if basis is not None:
        r2s = []
        for j in range(basis.shape[1]):
            v = basis[:, j]
            t, p = truth @ v, pred @ v
            r2s.append(float(1 - ((p - t) ** 2).mean() / max(t.var(), 1e-12)))
        out["dir_r2"] = [round(v, 4) for v in r2s]
        out["dir_r2_mean"] = float(np.mean(r2s))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-groups", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="v3a")
    parser.add_argument("--train-splits", default="train,train_aug")
    parser.add_argument("--w-n", type=float, default=0.5)
    parser.add_argument("--w-nll", type=float, default=1.0)
    parser.add_argument("--w-emit", type=float, default=0.3)
    parser.add_argument("--w-state", type=float, default=0.5)
    parser.add_argument("--w-rate", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train = load_concat(args.train_splits)
    val = load("val")
    fisher_path = RESULTS / "fisher_synth_v2.json"
    if fisher_path.exists():
        F = np.asarray(json.loads(fisher_path.read_text())["fisher"])
    else:
        F = compute_fisher(B2.load_split("val"))
        RESULTS.mkdir(parents=True, exist_ok=True)
        fisher_path.write_text(json.dumps({"fisher": F.tolist()}, indent=2))
    evals, evecs = np.linalg.eigh(F)
    keep = evals > 0.01 * evals.max()
    W = (evecs[:, keep] * evals[keep]) @ evecs[:, keep].T
    W = W / np.trace(W) * len(B2.RATE_INDEX)
    rate_metric = torch.as_tensor(W, dtype=torch.float32, device=V3.DEVICE)
    order = np.argsort(evals)[::-1][:4]
    basis = evecs[:, order]
    print(f"train: {len(train[0])} traces, {len(train[5])} groups | fisher keeps {int(keep.sum())}/12", flush=True)

    model = V3.KIHMMv3(hidden=args.hidden).to(V3.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    X, y, r, N, group, R, K = train
    best = -np.inf
    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        losses = []
        for gids in group_batches(len(R), args.batch_groups, rng):
            idx = (np.asarray(gids)[:, None] * K + np.arange(K)[None, :]).reshape(-1)
            xb, rb, Nb, Rb, gb = batch_tensors(X, r, N, R, K, gids)
            out = model(xb, gb, n_ref=Nb)
            loss, _ = V3.kihmm_v3_loss(out, Nb, torch.as_tensor(y[idx], device=V3.DEVICE),
                                       T2.counts_from_r(rb), Rb[gb],
                                       w_n=args.w_n, w_nll=args.w_nll, w_emit=args.w_emit,
                                       w_state=args.w_state, w_rate=args.w_rate,
                                       rate_metric=rate_metric)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        vm = evaluate_v3(model, val, basis=basis)
        score = vm["open_acc"] + 0.3 * vm["dir_r2_mean"] - 0.05 * vm["state_mae"]
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **vm, "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(f"[{epoch:02d}] loss={row['loss']:.3f} N={vm['n_acc']:.3f} "
              f"open={vm['open_acc']:.4f} open_mae={vm['open_mae']:.3f} "
              f"state_mae={vm['state_mae']:.3f} dirR2={np.round(vm['dir_r2'], 3)} "
              f"({row['seconds']}s)", flush=True)
        if score > best:
            best = score
            MODELS.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                       MODELS / f"kihmm_v3_{args.tag}.pt")

    ckpt = torch.load(MODELS / f"kihmm_v3_{args.tag}.pt", map_location=V3.DEVICE)
    model.load_state_dict(ckpt["model"])
    report = {"history": history, "final": {"val": evaluate_v3(model, val, basis=basis)}, "test": {}, "extrap": {}}
    for s in ["1", "2", "4"]:
        d = B2.load_split("test")
        base = load("test")
        data = (d[f"X_s{s}"], base[1], base[2], base[3], base[4], base[5], base[6])
        report["test"][f"scale_{s}"] = evaluate_v3(model, data, basis=basis)
        print(f"test x{s}: {report['test'][f'scale_{s}']}", flush=True)
    ex = load("extrap")
    for s in ["1", "2"]:
        d = B2.load_split("extrap")
        data = (d[f"X_s{s}"], ex[1], ex[2], ex[3], ex[4], ex[5], ex[6])
        report["extrap"][f"scale_{s}"] = evaluate_v3(model, data, basis=basis)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"kihmm_v3_{args.tag}.json").write_text(json.dumps(report, indent=2))
    print("saved", MODELS / f"kihmm_v3_{args.tag}.pt", flush=True)


if __name__ == "__main__":
    main()

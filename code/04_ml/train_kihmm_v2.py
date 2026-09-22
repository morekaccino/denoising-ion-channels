"""Train KI-HMM v2 on the rate-randomized benchmark (synth_v2).

Artifacts:
  code/04_ml/models/kihmm_v2_synth_v2.pt
  code/04_ml/results/kihmm_v2_synth_v2.json

Usage:
  python code/04_ml/train_kihmm_v2.py --epochs 60 --batch-groups 8
  python code/04_ml/train_kihmm_v2.py --mini --epochs 2 --batch-groups 4
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from scipy.linalg import expm

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "04_ml"))

import benchmark as B  # noqa: E402
import benchmark_v2 as B2  # noqa: E402
import kinetics as K  # noqa: E402
import torch_models_v2 as T  # noqa: E402

MODELS = ROOT / "code" / "04_ml" / "models"
RESULTS = ROOT / "code" / "04_ml" / "results"


def load(name: str = "train"):
    d = B2.load_split(name)
    X = d["X"] if "X" in d else d["X_s1"]
    K = int((d["group"] == 0).sum())
    return X, d["y"], d["r"], d["N"], d["group"], d["R"], K


def load_concat(names: str):
    parts = [load(n.strip()) for n in names.split(",") if n.strip()]
    Xs, ys, rs, Ns, gs, Rs = [], [], [], [], [], []
    offset = 0
    for X, y, r, N, g, R, K in parts:
        Xs.append(X)
        ys.append(y)
        rs.append(r)
        Ns.append(N)
        gs.append(g + offset)
        Rs.append(R)
        offset += len(R)
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(rs),
            np.concatenate(Ns), np.concatenate(gs), np.concatenate(Rs), parts[0][6])


def batch_tensors(X, r, N, R, K, gids, k_use=None):
    ku = K if k_use is None else k_use
    idx = (np.asarray(gids)[:, None] * K + np.arange(ku)[None, :]).reshape(-1)
    xb = torch.as_tensor(X[idx], dtype=torch.float32, device=T.DEVICE)
    rb = torch.as_tensor(r[idx].astype(np.int64), device=T.DEVICE)
    Nb = torch.as_tensor(N[idx].astype(np.int64), device=T.DEVICE)
    Rb = torch.as_tensor(R[gids], dtype=torch.float32, device=T.DEVICE)
    gb = torch.arange(len(gids), device=T.DEVICE).repeat_interleave(ku)
    return xb, rb, Nb, Rb, gb


def group_batches(n_groups: int, per_batch: int, rng: np.random.Generator, shuffle: bool = True):
    order = rng.permutation(n_groups) if shuffle else np.arange(n_groups)
    for s in range(0, n_groups, per_batch):
        yield order[s:s + per_batch]


def compute_fisher(d, traces_per_n: int = 12, eps: float = 1e-2) -> np.ndarray:
    """Fisher information of log p(signal | rates), exact decoder, per trace.

    Only some directions of the 12-rate vector are identifiable from summed
    traces; this matrix tells the loss which ones to weight.
    """
    F = np.zeros((len(B2.RATE_INDEX),) * 2)
    n_used = 0
    for N0 in (1, 2, 3):
        idx = np.flatnonzero(d["N"] == N0)[:traces_per_n]
        grid = K.default_grid(N0)
        logE = K.emission_log_densities(N0, grid)
        for i in idx:
            X = d["X_s1"][i]
            Rv = d["R"][d["group"][i]]

            def logZ(v):
                return K.decode_open_count(X, N0, P0=expm(B.DT * B2.rate_table(np.asarray(v))),
                                           grid=grid, logE=logE, return_evidence=True)[2]

            l0 = logZ(Rv)
            g = np.empty(len(Rv))
            for j in range(len(Rv)):
                v = Rv.copy()
                v[j] *= np.exp(eps)
                g[j] = (logZ(v) - l0) / eps
            F += np.outer(g, g)
            n_used += 1
    return F / max(n_used, 1)


@torch.no_grad()
def evaluate(model: T.KIHMMv2, data, per_batch: int = 16, basis: np.ndarray | None = None,
             k_use: int | None = None) -> dict:
    X, y, r, N, group, R, K = data
    ku = K if k_use is None else k_use
    T_steps = y.shape[1]
    model.eval()
    n_groups = len(R)
    n_ok = 0
    state_abs = 0.0
    open_ok = 0
    open_total = 0
    rate_abs = np.zeros(len(B2.RATE_INDEX))
    rate_log_abs = np.zeros(len(B2.RATE_INDEX))
    rate_rel = np.zeros(len(B2.RATE_INDEX))
    rate_count = 0
    rate_preds, rate_truths = [], []
    for gids in group_batches(n_groups, per_batch, np.random.default_rng(0), shuffle=False):
        xb, rb, Nb, Rb, gb = batch_tensors(X, r, N, R, K, gids, k_use=ku)
        out = T.predict(model, xb, gb)
        counts_true = T.counts_from_r(rb)
        n_ok += int((out["N_hat"] == Nb).sum())
        state_abs += float((out["counts"] - counts_true).abs().sum())
        open_pred = torch.round(T.state_open_counts(out["counts"]))
        open_ok += int((open_pred == torch.as_tensor(y, device=open_pred.device)[
            (np.asarray(gids)[:, None] * K + np.arange(ku)[None, :]).reshape(-1)]).sum())
        open_total += open_pred.numel()
        err = (out["rates"] - Rb).abs().cpu().numpy()
        log_err = (torch.log(out["rates"]) - torch.log(Rb)).abs().cpu().numpy()
        rate_abs += err.sum(axis=0)
        rate_log_abs += log_err.sum(axis=0)
        rate_rel += (err / np.maximum(Rb.cpu().numpy(), 1e-9)).sum(axis=0)
        rate_preds.append(np.log(out["rates"].cpu().numpy()))
        rate_truths.append(np.log(Rb.cpu().numpy()))
        rate_count += len(gids)
    n_traces = int(r.shape[0])
    pred, truth = np.concatenate(rate_preds), np.concatenate(rate_truths)
    result = {
        "n_acc": n_ok / (n_traces or 1),
        "state_mae": state_abs / (n_traces * T_steps * 7),
        "open_acc": open_ok / open_total,
        "rate_mae": float(rate_log_abs.sum() / (rate_count * len(B2.RATE_INDEX))),
        "rate_rel_mae": float(rate_rel.sum() / (rate_count * len(B2.RATE_INDEX))),
        "rate_rel_per_rate": (rate_rel / rate_count).round(4).tolist(),
        "rate_abs_per_rate": (rate_abs / rate_count).round(4).tolist(),
        "rate_r2_per_rate": [round(float(1 - ((pred[:, j] - truth[:, j]) ** 2).mean()
                                       / max(truth[:, j].var(), 1e-12)), 4)
                             for j in range(len(B2.RATE_INDEX))],
    }
    if basis is not None:
        dir_r2 = []
        for r in range(basis.shape[1]):
            v = basis[:, r]
            t, p = truth @ v, pred @ v
            dir_r2.append(float(1.0 - ((p - t) ** 2).mean() / max(t.var(), 1e-12)))
        result["dir_r2"] = [round(v, 4) for v in dir_r2]
        result["dir_r2_mean"] = float(np.mean(dir_r2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-groups", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--tag", default="synth_v2")
    parser.add_argument("--w-n", type=float, default=0.3)
    parser.add_argument("--w-state", type=float, default=1.0)
    parser.add_argument("--w-rate", type=float, default=1.0)
    parser.add_argument("--train-splits", default="train,train_aug")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train = load_concat(args.train_splits)
    val = load("val")
    print(f"train: {len(train[0])} traces, {len(train[5])} groups, K={train[6]}", flush=True)
    print(f"val:   {len(val[0])} traces, {len(val[5])} groups, K={val[6]}", flush=True)

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
    rate_metric = torch.as_tensor(W, dtype=torch.float32, device=T.DEVICE)
    order = np.argsort(evals)[::-1][:4]
    basis = evecs[:, order]
    print(f"fisher: keeping {int(keep.sum())}/{len(evals)} rate directions", flush=True)

    model = T.KIHMMv2(hidden=args.hidden).to(T.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    X, y, r, N, group, R, K = train
    best = -np.inf
    history = []
    suffix = "_mini" if args.mini else ""
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        losses = []
        for gids in group_batches(len(R), args.batch_groups, rng):
            xb, rb, Nb, Rb, gb = batch_tensors(X, r, N, R, K, gids)
            out = model(xb, gb)
            loss, _ = T.kihmm_v2_loss(out, Nb, T.counts_from_r(rb), Rb[gb],
                                      w_n=args.w_n, w_state=args.w_state, w_rate=args.w_rate,
                                      rate_metric=rate_metric)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        vm = evaluate(model, val, basis=basis)
        score = vm["n_acc"] - 0.1 * vm["state_mae"] + 0.3 * vm["dir_r2_mean"]
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_n_acc": vm["n_acc"],
            "val_state_mae": vm["state_mae"],
            "val_open_acc": vm["open_acc"],
            "val_rate_mae": vm["rate_mae"],
            "val_dir_r2": vm.get("dir_r2"),
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(
            f"[{epoch:02d}] loss={row['loss']:.3f} N={vm['n_acc']:.3f} "
            f"state_mae={vm['state_mae']:.3f} open={vm['open_acc']:.4f} "
            f"rate_mae={vm['rate_mae']:.3f} dirR2={np.round(vm['dir_r2'], 3)} ({row['seconds']}s)",
            flush=True,
        )
        if score > best:
            best = score
            MODELS.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                       MODELS / f"kihmm_v2_{args.tag}{suffix}.pt")

    ckpt = torch.load(MODELS / f"kihmm_v2_{args.tag}{suffix}.pt", map_location=T.DEVICE)
    model.load_state_dict(ckpt["model"])
    report = {"history": history, "final": {}, "test": {}, "extrap": {}}
    report["final"]["val"] = evaluate(model, val, basis=basis)
    test = load("test")
    for s in ["1", "2", "4"]:
        d = B2.load_split("test")
        Xs = d[f"X_s{s}"]
        data = (Xs, test[1], test[2], test[3], test[4], test[5], test[6])
        report["test"][f"scale_{s}"] = evaluate(model, data, basis=basis)
        print(f"test x{s}: { {k: round(v, 4) for k, v in report['test'][f'scale_{s}'].items() if isinstance(v, float)} }", flush=True)
    extrap = load("extrap")
    for s in ["1", "2"]:
        d = B2.load_split("extrap")
        data = (d[f"X_s{s}"], extrap[1], extrap[2], extrap[3], extrap[4], extrap[5], extrap[6])
        report["extrap"][f"scale_{s}"] = evaluate(model, data, basis=basis)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"kihmm_v2_{args.tag}{suffix}.json").write_text(json.dumps(report, indent=2))
    print("saved", MODELS / f"kihmm_v2_{args.tag}{suffix}.pt", flush=True)


if __name__ == "__main__":
    main()

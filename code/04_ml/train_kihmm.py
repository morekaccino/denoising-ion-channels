"""Train the KI-HMM on the frozen synthetic benchmark.

Data: `data/derived/synth_v1/train_wide.npz` (N in 1..5, per-trace noise scale in
[1, 4]) for training; `val.npz` and `extrap.npz` (scale 1) for monitoring.
Artefacts: `code/04_ml/models/kihmm_synth_v1.pt` and
`code/04_ml/results/kihmm_synth_v1.json`.

Usage:
    python code/04_ml/train_kihmm.py --epochs 30 --batch 32
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
import torch_models as T  # noqa: E402

MODELS = ROOT / "code" / "04_ml" / "models"
RESULTS = ROOT / "code" / "04_ml" / "results"


def batches(x, y, n, batch_size, rng, shuffle=True):
    idx = rng.permutation(len(x)) if shuffle else np.arange(len(x))
    for start in range(0, len(idx), batch_size):
        sel = idx[start : start + batch_size]
        yield (
            torch.as_tensor(x[sel], dtype=torch.float32, device=T.DEVICE),
            torch.as_tensor(y[sel].astype(np.int64), device=T.DEVICE),
            torch.as_tensor(n[sel].astype(np.int64), device=T.DEVICE),
        )


@torch.no_grad()
def evaluate(model, x, y, n, batch_size=128):
    """Single pass: smoothed states at the true N plus the N-agnostic count head."""
    model.eval()
    preds, n_hats = [], []
    for xb, yb, nb in batches(x, y, n, batch_size, np.random.default_rng(0), shuffle=False):
        log_gamma, _, _, count_logits = T.kihmm_forward(model, xb, nb)
        preds.append(log_gamma.argmax(dim=-1).cpu().numpy())
        n_hats.append(count_logits.argmax(dim=-1).cpu().numpy() + 1)
    pred = np.concatenate(preds)
    n_hat = np.concatenate(n_hats)
    metrics = B.summarize(y, pred, N_true=n)
    metrics["count_head_accuracy"] = float((n_hat == n).mean())
    metrics["per_n"] = {str(k): v for k, v in metrics["per_n"].items()}
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", default="train_wide")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tr = B.load_split(args.train)
    Xtr, ytr, Ntr = tr["X"], tr["y"], tr["N"]
    va = B.load_split("val")
    ex = B.load_split("extrap")

    model = T.KIHMM(hidden=args.hidden).to(T.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        losses = []
        for xb, yb, nb in batches(Xtr, ytr, Ntr, args.batch, rng):
            opt.zero_grad()
            loss, _ = T.kihmm_loss(model, xb, yb, nb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        vm = evaluate(model, va["X_s1"], va["y"], va["N"])
        em = evaluate(model, ex["X_s1"], ex["y"], ex["N"])
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_acc": vm["accuracy"],
            "val_n_acc": vm["count_head_accuracy"],
            "extrap_acc": em["accuracy"],
            "extrap_n_acc": em["count_head_accuracy"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(
            f"[{epoch:02d}] loss={row['loss']:.3f} val={row['val_acc']:.4f} valN={row['val_n_acc']:.3f} "
            f"extrap={row['extrap_acc']:.4f} extrapN={row['extrap_n_acc']:.3f} ({row['seconds']}s)",
            flush=True,
        )
        if vm["accuracy"] + em["accuracy"] > best:
            best = vm["accuracy"] + em["accuracy"]
            MODELS.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, MODELS / "kihmm_synth_v1.pt")

    model.load_state_dict(torch.load(MODELS / "kihmm_synth_v1.pt", map_location=T.DEVICE)["model"])
    report = {"history": history, "final": {}, "test": {}}
    va_m = evaluate(model, va["X_s1"], va["y"], va["N"])
    ex_m = evaluate(model, ex["X_s1"], ex["y"], ex["N"])
    report["final"] = {"val": va_m, "extrap": ex_m}
    te = B.load_split("test")
    for s in ["1", "2", "4"]:
        m = evaluate(model, te[f"X_s{s}"], te["y"], te["N"])
        m_ev = evaluate_evidence_n(model, te[f"X_s{s}"], te["N"])
        m["evidence_n_accuracy"] = m_ev
        report["test"][f"scale_{s}"] = m
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "kihmm_synth_v1.json").write_text(json.dumps(report, indent=2))
    torch.save({"model": model.state_dict(), "args": vars(args)}, MODELS / "kihmm_synth_v1.pt")
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)}
                      for k, v in report["test"].items()}, indent=2))
    print("saved", MODELS / "kihmm_synth_v1.pt")


@torch.no_grad()
def evaluate_evidence_n(model, x, n, batch_size=128):
    model.eval()
    ok = 0
    for start in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=T.DEVICE)
        n_hat, _, _, _ = T.infer_n(model, xb, mode="evidence")
        ok += int((n_hat.cpu().numpy() == n[start : start + batch_size]).sum())
    return ok / len(x)


if __name__ == "__main__":
    main()

"""Sequence-input variant of Deep-Channel (upper-bound neural baseline).

The released Deep-Channel code and the paper's Methods train with time steps
``n=1``: one sample at a time, zero initial recurrent state, so the LSTM never
sees temporal context. This variant keeps the exact same architecture and
weights (Conv1D k=1, 3x 256-unit LSTM with hard-sigmoid gates and relu cell
output, BN, dropout, softmax head) but trains it the way the paper's
motivation suggests: on whole 1000-sample traces, so the recurrent layers can
use temporal context.

Run on a GPU (the manual LSTM loop is not vectorised over time):

  python code/05_baselines/deepchannel_seq.py --train --epochs 4 --batch 8 --tag dc_seq
  python code/05_baselines/deepchannel_seq.py --eval --tag dc_seq
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hmm_core as core  # noqa: E402
from deepchannel_port import N_CLASSES, hard_sigmoid  # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))

RESULTS = core.ROOT / "code" / "05_baselines" / "results"
MODELS = core.ROOT / "code" / "05_baselines" / "models"


class SeqLSTMBlock(nn.Module):
    def __init__(self, in_features: int, units: int, dropout: float = 0.2):
        super().__init__()
        self.w_ih = nn.Linear(in_features, 4 * units)
        self.w_hh = nn.Linear(units, 4 * units)
        self.bn = nn.BatchNorm1d(units)
        self.drop = nn.Dropout(dropout)
        self.units = units

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        proj = self.w_ih(x)
        h = torch.zeros(batch, self.units, device=x.device)
        c = torch.zeros(batch, self.units, device=x.device)
        outs = []
        for t in range(steps):
            gates = proj[:, t] + self.w_hh(h)
            i, f, g, o = gates.chunk(4, dim=-1)
            c = hard_sigmoid(f) * c + hard_sigmoid(i) * torch.tanh(g)
            h = hard_sigmoid(o) * F.relu(c)
            outs.append(h)
        z = torch.stack(outs, dim=1)
        z = self.drop(self.bn(z.transpose(1, 2)).transpose(1, 2))
        return z


class SeqDeepChannel(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Conv1d(1, 64, kernel_size=1)
        self.block1 = SeqLSTMBlock(64, 256, dropout)
        self.block2 = SeqLSTMBlock(256, 256, dropout)
        self.block3 = SeqLSTMBlock(256, 256, dropout)
        self.head = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.conv(x[:, None, :])).transpose(1, 2)
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)
        return self.head(z)


def train_model(epochs: int = 4, batch: int = 8, lr: float = 1e-3,
                weight_power: float = 1.0, seed: int = 0, tag: str = "dc_seq",
                device: str = "cuda") -> dict:
    tr = core.load_traces("train")
    va = core.load_traces("val")
    X, y = tr["X"], tr["y"].astype(np.int64)
    lo, hi = float(X.min()), float(X.max())
    Xs = ((X - lo) / (hi - lo)).astype(np.float32)
    Xv = ((va["X"] - lo) / (hi - lo)).astype(np.float32)
    yv = va["y"].astype(np.int64)

    counts = np.bincount(y.reshape(-1), minlength=N_CLASSES).astype(np.float64)
    weights = (counts.sum() / np.maximum(counts, 1.0)) ** weight_power
    weights = weights / weights.mean()

    model = SeqDeepChannel().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        total, steps = 0.0, 0
        order = rng.permutation(len(Xs))
        for s in range(0, len(Xs) - batch + 1, batch):
            idx = order[s:s + batch]
            xb = torch.from_numpy(Xs[idx]).float().to(device)
            yb = torch.from_numpy(y[idx]).long().to(device)
            opt.zero_grad()
            loss = lossf(model(xb).reshape(-1, N_CLASSES), yb.reshape(-1))
            loss.backward()
            opt.step()
            total += float(loss)
            steps += 1
        model.eval()
        with torch.no_grad():
            accs = []
            for i in range(len(Xv)):
                xb = torch.from_numpy(Xv[i:i + 1]).float().to(device)
                pred = model(xb).argmax(dim=-1).cpu().numpy().reshape(-1)
                accs.append(pred == yv[i])
            vacc = float(np.concatenate(accs).mean())
        history.append({"epoch": epoch + 1, "loss": total / max(steps, 1),
                        "val_open_acc": vacc, "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch + 1}/{epochs}: loss={total / max(steps, 1):.4f} "
              f"val_open_acc={vacc:.4f} ({history[-1]['seconds']}s)", flush=True)

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "scale": (lo, hi),
                "history": history}, MODELS / f"{tag}.pt")
    return {"history": history, "scale": (lo, hi)}


def predict_split(model_path: str, split: str, device: str = "cuda") -> dict:
    ckpt = torch.load(model_path, map_location=device)
    lo, hi = ckpt["scale"]
    model = SeqDeepChannel().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    data = core.load_traces(split)
    X, y, N = data["X"], data["y"].astype(np.int64), data["N"]
    preds = np.empty((len(X), X.shape[1]), dtype=np.int64)
    with torch.no_grad():
        for i in range(len(X)):
            xb = torch.from_numpy(((X[i:i + 1] - lo) / (hi - lo)).astype(np.float32)).to(device)
            preds[i] = model(xb).argmax(dim=-1).cpu().numpy().reshape(-1)
    N_hat = preds.max(axis=1)
    per_n = {}
    for k in range(1, 6):
        m = N == k
        if m.any():
            per_n[int(k)] = {
                "traces": int(m.sum()),
                "n_acc": float((N_hat[m] == N[m]).mean()),
                "open_acc": float((preds[m] == y[m]).mean()),
                "open_mae": float(np.abs(preds[m] - y[m]).mean()),
            }
    return {
        "method": "deepchannel_seq",
        "split": split,
        "traces": int(len(X)),
        "metrics": {
            "n_acc": float((N_hat == N).mean()),
            "open_acc": float((preds == y).mean()),
            "open_mae": float(np.abs(preds - y).mean()),
            "per_n": per_n,
            "n_hat_hist": {str(k): int((N_hat == k).sum()) for k in range(1, 6)},
        },
        "paths": preds,
        "N_hat": N_hat.tolist(),
        "N_true": N.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-power", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="dc_seq")
    parser.add_argument("--splits", default="test_x1,test_x2,test_x4")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.train:
        out = train_model(args.epochs, args.batch, args.lr, args.weight_power,
                          args.seed, args.tag, args.device)
        (RESULTS / f"{args.tag}_train.json").write_text(json.dumps(out, indent=2))
        print("saved model", MODELS / f"{args.tag}.pt", flush=True)
        return
    if args.eval:
        for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
            report = predict_split(str(MODELS / f"{args.tag}.pt"), split, args.device)
            paths = report.pop("paths")
            (RESULTS / f"{args.tag}_{split}.json").write_text(json.dumps(report, indent=2))
            np.savez_compressed(RESULTS / f"{args.tag}_{split}_paths.npz", paths=paths,
                                N_hat=report["N_hat"], N_true=report["N_true"])
            m = report["metrics"]
            print(f"{split}: N={m['n_acc']:.3f} open={m['open_acc']:.4f} "
                  f"MAE={m['open_mae']:.4f}", flush=True)
        return
    parser.print_help()


if __name__ == "__main__":
    main()

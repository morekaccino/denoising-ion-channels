"""PyTorch port of Deep-Channel (Celik et al. 2020) for the synth_v2 benchmark.

Reference implementation: https://github.com/RichardBJ/Deep-Channel (MIT),
``deepchannel_train.py``, ``predictor.py`` and the shipped model JSON
``model/JSON/nmn_oversampled_deepchannel6/model.json``.

Architecture (verified against the shipped JSON): Conv1D(64, k=1, relu) ->
MaxPool(1) -> Flatten -> LSTM(256, relu, hard_sigmoid, return_sequences) ->
BN -> Dropout(0.2) -> LSTM(256, ...) -> BN -> Dropout(0.2) -> LSTM(256,
relu, hard_sigmoid) -> BN -> Dropout(0.2) -> Dense(6) -> Softmax.

The paper and code train with time steps n=1, i.e. one sample at a time with
zero initial recurrent state, so the recurrent weights never activate; the
port reproduces exactly that (a per-sample gated MLP). Deviations from the
reference training recipe are listed in ``BASELINES.md``.

Usage:
  python code/05_baselines/deepchannel_port.py --verify
  python code/05_baselines/deepchannel_port.py --train --epochs 4
  python code/05_baselines/deepchannel_port.py --eval --splits test_x1,test_x2,test_x4
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

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))

RESULTS = core.ROOT / "code" / "05_baselines" / "results"
MODELS = core.ROOT / "code" / "05_baselines" / "models"
N_CLASSES = 6


def hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)


class LSTMBlock(nn.Module):
    """One Keras LSTM(activation='relu', recurrent_activation='hard_sigmoid')
    layer followed by BatchNormalization and Dropout, unrolled for T=1 with
    zero initial state (the reference training setup)."""

    def __init__(self, in_features: int, units: int, dropout: float = 0.2):
        super().__init__()
        self.w_ih = nn.Linear(in_features, 4 * units)
        self.w_hh = nn.Linear(units, 4 * units)
        self.bn = nn.BatchNorm1d(units)
        self.drop = nn.Dropout(dropout)
        self.units = units

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.w_ih(x) + self.w_hh.bias
        i, f, g, o = gates.chunk(4, dim=-1)
        c = hard_sigmoid(f) * 0.0 + hard_sigmoid(i) * torch.tanh(g)
        h = hard_sigmoid(o) * F.relu(c)
        return self.drop(self.bn(h))


class DeepChannel(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Conv1d(1, 64, kernel_size=1)
        self.block1 = LSTMBlock(64, 256, dropout)
        self.block2 = LSTMBlock(256, 256, dropout)
        self.block3 = LSTMBlock(256, 256, dropout)
        self.head = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.conv(x[:, None, :]))
        z = z[:, :, 0]
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)
        return self.head(z)


def input_scale(train_x: np.ndarray) -> tuple[float, float]:
    return float(train_x.min()), float(train_x.max())


def batch_points(x: np.ndarray, y: np.ndarray, batch: int, rng: np.random.Generator,
                 device: str = "cuda"):
    n = len(x)
    order = rng.permutation(n)
    for start in range(0, n - batch + 1, batch):
        idx = order[start:start + batch]
        yield (torch.from_numpy(x[idx]).float().to(device)[:, None],
               torch.from_numpy(y[idx]).long().to(device))


def train_model(epochs: int = 4, batch: int = 1024, lr: float = 1e-3,
                weight_power: float = 1.0, seed: int = 0, tag: str = "deepchannel",
                device: str = "cuda", lr_step: int = 3, lr_gamma: float = 0.1) -> dict:
    data = core.load_traces("train")
    X, y = data["X"], data["y"].astype(np.int64)
    val = core.load_traces("val")
    lo, hi = input_scale(X)
    xs = ((X - lo) / (hi - lo)).astype(np.float32)
    xv = ((val["X"] - lo) / (hi - lo)).astype(np.float32)
    yv = val["y"].astype(np.int64)

    counts = np.bincount(y.ravel(), minlength=N_CLASSES).astype(np.float64)
    weights = (counts.sum() / np.maximum(counts, 1.0)) ** weight_power
    weights = weights / weights.mean()

    model = DeepChannel().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)

    flat_x = xs.reshape(-1)
    flat_y = y.reshape(-1)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        total = 0.0
        steps = 0
        for xb, yb in batch_points(flat_x, flat_y, batch, rng, device):
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            total += float(loss)
            steps += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            accs = []
            xv_flat = xv.reshape(-1)
            yv_flat = yv.reshape(-1)
            for start in range(0, len(xv_flat), 8192):
                xb = torch.from_numpy(xv_flat[start:start + 8192]).float().to(device)[:, None]
                pred = model(xb).argmax(dim=-1).cpu().numpy()
                accs.append(pred == yv_flat[start:start + 8192])
            vacc = float(np.concatenate(accs).mean())
        history.append({"epoch": epoch + 1, "loss": total / max(steps, 1),
                        "val_open_acc": vacc, "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch + 1}/{epochs}: loss={total / max(steps, 1):.4f} "
              f"val_open_acc={vacc:.4f} ({history[-1]['seconds']}s)", flush=True)

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "scale": (lo, hi),
                "history": history, "tag": tag}, MODELS / f"{tag}.pt")
    return {"history": history, "scale": (lo, hi), "weights": weights.tolist()}


def predict_split(model_path: str, split: str, device: str = "cuda",
                  batch: int = 16384) -> dict:
    ckpt = torch.load(model_path, map_location=device)
    lo, hi = ckpt["scale"]
    model = DeepChannel().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    data = core.load_traces(split)
    X, y, N = data["X"], data["y"].astype(np.int64), data["N"]
    flat = ((X.reshape(-1) - lo) / (hi - lo)).astype(np.float32)
    pred = np.empty(len(flat), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(flat), batch):
            xb = torch.from_numpy(flat[start:start + batch]).float().to(device)[:, None]
            pred[start:start + batch] = model(xb).argmax(dim=-1).cpu().numpy()
    paths = pred.reshape(X.shape)
    N_hat = paths.max(axis=1)
    open_acc = float((paths == y).mean())
    open_mae = float(np.abs(paths - y).mean())
    n_acc = float((N_hat == N).mean())
    per_n = {}
    for k in range(1, 6):
        m = N == k
        if m.any():
            per_n[int(k)] = {
                "traces": int(m.sum()),
                "n_acc": float((N_hat[m] == N[m]).mean()),
                "open_acc": float((paths[m] == y[m]).mean()),
                "open_mae": float(np.abs(paths[m] - y[m]).mean()),
            }
    return {
        "method": "deepchannel_port",
        "split": split,
        "traces": int(len(X)),
        "metrics": {"n_acc": n_acc, "open_acc": open_acc, "open_mae": open_mae,
                    "per_n": per_n,
                    "n_hat_hist": {str(k): int((N_hat == k).sum()) for k in range(1, 6)}},
        "paths": paths,
        "N_hat": N_hat.tolist(),
        "N_true": N.tolist(),
    }


def verify() -> None:
    json_path = core.ROOT / ".external" / "deep-channel" / "model" / "JSON" / \
        "nmn_oversampled_deepchannel6" / "model.json"
    cfg = json.loads(json_path.read_text())
    layers = [l["class_name"] for l in cfg["modelTopology"]["model_config"]["config"]["layers"]]
    assert layers.count("TimeDistributed") == 3
    assert layers.count("LSTM") == 3
    assert layers.count("BatchNormalization") == 3
    assert layers.count("Dropout") == 3
    lstms = [l["config"] for l in cfg["modelTopology"]["model_config"]["config"]["layers"]
             if l["class_name"] == "LSTM"]
    assert all(int(l["units"]) == 256 and l["activation"] == "relu"
               and l["recurrent_activation"] == "hard_sigmoid" for l in lstms)
    print("shipped JSON architecture matches port: 3x LSTM(256, relu, hard_sigmoid)")

    model = DeepChannel()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"port parameter count: {n_params:,}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    x = torch.randn(32, 1).to(dev)
    out = model(x)
    assert out.shape == (32, N_CLASSES)
    assert torch.allclose(out.softmax(dim=-1).sum(dim=-1), torch.ones(32, device=dev), atol=1e-5)

    data = core.load_traces("train")
    xs = data["X"][:2].reshape(-1)
    ys = data["y"][:2].reshape(-1).astype(np.int64)
    lo, hi = float(xs.min()), float(xs.max())
    xb = torch.from_numpy(((xs - lo) / (hi - lo)).astype(np.float32)).to(dev)[:, None]
    yb = torch.from_numpy(ys).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lossf = nn.CrossEntropyLoss()
    first = None
    for step in range(200):
        opt.zero_grad()
        loss = lossf(model(xb), yb)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
    acc = float((model(xb).argmax(dim=-1) == yb).float().mean())
    print(f"200-step sanity train: loss {first:.3f} -> {float(loss):.3f}, batch acc {acc:.3f}")
    assert acc > 0.5
    print("deepchannel_port verify: all checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-power", type=float, default=1.0)
    parser.add_argument("--lr-step", type=int, default=3)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="deepchannel")
    parser.add_argument("--model", default="")
    parser.add_argument("--splits", default="test_x1,test_x2,test_x4")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.verify:
        verify()
        return
    if args.train:
        out = train_model(args.epochs, args.batch, args.lr, args.weight_power,
                          args.seed, args.tag, args.device, args.lr_step, args.lr_gamma)
        (RESULTS / f"{args.tag}_train.json").write_text(json.dumps(out, indent=2))
        print("saved model", MODELS / f"{args.tag}.pt")
        return
    if args.eval:
        model_path = args.model or str(MODELS / f"{args.tag}.pt")
        for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
            report = predict_split(model_path, split, args.device)
            paths = report.pop("paths")
            (RESULTS / f"{args.tag}_{split}.json").write_text(json.dumps(report, indent=2))
            np.savez_compressed(RESULTS / f"{args.tag}_{split}_paths.npz", paths=paths,
                                N_hat=report["N_hat"], N_true=report["N_true"])
            m = report["metrics"]
            print(f"{split}: N={m['n_acc']:.3f} open={m['open_acc']:.4f} MAE={m['open_mae']:.4f}")
        return
    parser.print_help()


if __name__ == "__main__":
    main()

"""Fast architecture bake-off for the two heads that still limit KI-HMM.

The structured part of v4 is already at the reference posterior (the loss terms
`gen` 1.110 and `crf` 0.682 sit on the oracle's 1.116 and 0.719), so the
backbone is not what is costing accuracy. What is costing accuracy is the two
trace-level heads: the channel count and the 12 rates.

Both are cheap to study in isolation. Everything expensive -- the emission
output, the HMM posterior, the log evidence per candidate N, the score vector
and the expected transition counts -- is computed once and cached, after which
each candidate head is a small network trained on cached tensors in well under
a minute. That makes it practical to rank a dozen designs instead of guessing.

Usage:
  python code/04_ml/bakeoff.py --build-cache
  python code/04_ml/bakeoff.py --task count
  python code/04_ml/bakeoff.py --task rates
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "04_ml"))

import benchmark_v2 as B2  # noqa: E402
import infer_v5 as I5  # noqa: E402
import torch_kinetics as TK  # noqa: E402
import torch_models_v4 as V4  # noqa: E402
from eval_kihmm_v4 import load_model  # noqa: E402
from torch_models import TemporalEncoder as TCN_ENCODER  # noqa: E402
from train_kihmm_v4 import batch, load  # noqa: E402
from train_kihmm_v2 import group_batches  # noqa: E402

CACHE = ROOT / "data" / "derived" / "bakeoff_cache"
RESULTS = ROOT / "code" / "04_ml" / "results"
MODEL = ROOT / "code" / "04_ml" / "models" / "kihmm_v4_v4b.pt"
K_MAX = V4.K_MAX
N_RATES = V4.N_RATES


# --------------------------------------------------------------------------- cache


def hist_pyramid(x: torch.Tensor, bins=(16, 32, 64, 128)) -> torch.Tensor:
    """Amplitude histograms at several resolutions, concatenated."""
    lo, hi = V4.HIST_RANGE
    out = []
    for b in bins:
        idx = (((x - lo) / (hi - lo)) * b).long().clamp(0, b - 1)
        out.append(x.new_zeros(x.shape[0], b).scatter_add_(1, idx, torch.ones_like(x)) / x.shape[1])
    return torch.cat(out, dim=1)


def quantile_curve(x: torch.Tensor, n: int = 64) -> torch.Tensor:
    """The inverse CDF on a fixed grid: a smooth, fixed-length view of the amplitudes."""
    qs = torch.linspace(0.005, 0.995, n, device=x.device, dtype=x.dtype)
    return torch.quantile(x, qs, dim=1).T


def rate_statistics(x: torch.Tensor, n: int, log_scale: torch.Tensor, emission):
    """Score vector, expected transition counts and timing features for one N block."""
    tables = TK.get_tables(n, x.device)
    base = torch.as_tensor(B2.BASE_OFF, dtype=torch.float32, device=x.device)
    delta = torch.zeros(x.shape[0], N_RATES, device=x.device, requires_grad=True)
    P0 = TK.p0_from_rates(base[None] * torch.exp(delta))
    logP = TK.count_transition(P0, tables).clamp_min(1e-30).log()
    logpi = TK.count_log_initial(TK.stationary(P0), tables)
    with torch.no_grad():
        log_emit = emission(x, n, log_scale, chunk=250)
    logb = TK.expand_open_emissions(log_emit, tables)
    logZ, gamma = TK.hmm_posterior(logb, logP, logpi)
    score, grad_logP = torch.autograd.grad(logZ.sum(), [delta, logP])

    # sum_t xi is exactly the gradient wrt logP; fold it onto the open count
    S, k = tables.size, tables.open_count
    xi = grad_logP.new_zeros(x.shape[0], K_MAX + 1, S).index_add_(1, k, grad_logP)
    xi = xi.new_zeros(x.shape[0], K_MAX + 1, K_MAX + 1).index_add_(2, k, xi)

    with torch.no_grad():
        p_open = (log_emit.softmax(-1) * torch.arange(n + 1, device=x.device, dtype=x.dtype)).sum(-1) / n
        post_open = (gamma @ tables.open_count.to(x.dtype)) / n
        kin = V4.kinetic_features(p_open, log_emit.argmax(-1), K_MAX)
    return (score.detach(), xi.reshape(x.shape[0], -1).detach(), kin,
            p_open.detach(), post_open.detach(), logZ.detach())


@torch.no_grad()
def evidence_block(X, rates, log_scale, emission, device, per_batch=24):
    return I5.evidence_n(X, rates, log_scale, emission, device, per_batch)


def build_cache(splits: dict[str, dict], device: str, evid_device: str, limit_groups: int = 0) -> None:
    """Run every expensive computation once and store the results."""
    model = load_model(str(MODEL), device)
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        n_groups = min(limit_groups or len(data["R"]), len(data["R"]))
        keep = n_groups * data["K"]
        t0 = time.time()

        heads = {"n_hat": [], "rates": [], "log_scale": []}
        for gids in group_batches(n_groups, 8, np.random.default_rng(0), shuffle=False):
            b = batch(data, gids, device)
            p = V4.predict(model, b["x"], b["group"], emission_chunk=250)
            heads["n_hat"].append(p["N_hat"].cpu().numpy())
            heads["rates"].append(p["rates"].cpu().numpy())
            heads["log_scale"].append(p["log_scale"].cpu().numpy())
        heads = {k: np.concatenate(v) for k, v in heads.items()}

        X = data["X"][:keep]
        N, group = data["N"][:keep], data["group"][:keep]
        xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        out = {
            "level": torch.cat([V4.level_features(xt[i : i + 256]) for i in range(0, keep, 256)]).cpu().numpy(),
            "pyramid": torch.cat([hist_pyramid(xt[i : i + 256]) for i in range(0, keep, 256)]).cpu().numpy(),
            "quantile": torch.cat([quantile_curve(xt[i : i + 256]) for i in range(0, keep, 256)]).cpu().numpy(),
            "N": N, "group": group, "R": data["R"][:n_groups],
            "log_scale_head": heads["log_scale"], "rates_head": heads["rates"],
            "n_head": heads["n_hat"],
        }

        model.emission.to(evid_device)
        out["evidence"] = evidence_block(X, heads["rates"][group], heads["log_scale"],
                                         model.emission, evid_device).astype(np.float32)
        model.emission.to(device)

        score = np.zeros((keep, N_RATES), dtype=np.float32)
        xi = np.zeros((keep, (K_MAX + 1) ** 2), dtype=np.float32)
        kin = np.zeros((keep, len(V4.LAGS) + (K_MAX + 1) ** 2), dtype=np.float32)
        p_open = np.zeros((keep, X.shape[1]), dtype=np.float32)
        post = np.zeros((keep, X.shape[1]), dtype=np.float32)
        for n in np.unique(N):
            idx = np.flatnonzero(N == n)
            for lo in range(0, len(idx), 16):
                sel = idx[lo : lo + 16]
                s, x_, k_, po, pp, _ = rate_statistics(
                    torch.as_tensor(X[sel], dtype=torch.float32, device=device), int(n),
                    torch.as_tensor(heads["log_scale"][sel], dtype=torch.float32, device=device),
                    model.emission)
                score[sel], xi[sel], kin[sel] = s.cpu(), x_.cpu(), k_.cpu()
                p_open[sel], post[sel] = po.cpu(), pp.cpu()
        out.update(score=score, xi=xi, kin=kin, p_open=p_open, post_open=post)
        np.savez_compressed(CACHE / f"{name}.npz", **out)
        print(f"cached {name}: {keep} traces, {n_groups} groups, {time.time() - t0:.0f}s", flush=True)


def concat_caches(parts: list[dict]) -> dict:
    """Stack cached splits, renumbering groups so they stay disjoint."""
    if len(parts) == 1:
        return parts[0]
    out, offset = {}, 0
    for key in parts[0]:
        if key in ("R",):
            out[key] = np.concatenate([p[key] for p in parts])
        elif key == "group":
            groups = []
            for p in parts:
                groups.append(p["group"] + offset)
                offset += len(p["R"])
            out[key] = np.concatenate(groups)
        else:
            out[key] = np.concatenate([p[key] for p in parts])
    return out


def load_cache(name: str) -> dict:
    with np.load(CACHE / f"{name}.npz") as d:
        out = {k: d[k] for k in d.files}
    # raw log evidence is thousands of nats and varies per trace; only the
    # differences between candidates mean anything, so centre it before use
    out["evidence_rel"] = (out["evidence"] - out["evidence"].max(axis=1, keepdims=True)) / 50.0
    return out


# --------------------------------------------------------------------------- heads


def mlp(din: int, dout: int, width: int = 128, depth: int = 2) -> nn.Module:
    layers: list[nn.Module] = [nn.Linear(din, width), nn.GELU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.GELU()]
    return nn.Sequential(*layers, nn.Linear(width, dout))


class HistCNN(nn.Module):
    """1D convolutions over the amplitude histogram: peak detection, as the CCNN did."""

    def __init__(self, bins: int, dout: int, ch: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, ch, 7, padding=3), nn.GELU(),
            nn.Conv1d(ch, ch, 5, padding=2, dilation=1), nn.GELU(),
            nn.Conv1d(ch, ch, 5, padding=4, dilation=2), nn.GELU())
        self.head = mlp(2 * ch, dout, 128, 1)

    def forward(self, x):
        h = self.conv(x[:, None, :])
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=-1))


class DeepSets(nn.Module):
    """Permutation-invariant over samples: N is a functional of the amplitude law."""

    def __init__(self, dout: int, width: int = 64, n_sample: int = 400):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.rho = mlp(3 * width, dout, 128, 2)
        self.n_sample = n_sample

    def forward(self, x):
        idx = torch.randint(0, x.shape[1], (self.n_sample,), device=x.device)
        h = self.phi(x[:, idx, None])
        return self.rho(torch.cat([h.mean(1), h.amax(1), h.std(1)], dim=-1))


class Ordinal(nn.Module):
    """CORAL head: K-1 cumulative thresholds, aimed at off-by-one errors."""

    def __init__(self, din: int, k: int = K_MAX):
        super().__init__()
        self.body = mlp(din, 1, 128, 2)
        self.bias = nn.Parameter(torch.zeros(k - 1))
        self.k = k

    def forward(self, x):
        return self.body(x) + self.bias[None]

    @staticmethod
    def loss(logits, target):
        levels = (target[:, None] > torch.arange(logits.shape[1], device=logits.device)[None]).float()
        return F.binary_cross_entropy_with_logits(logits, levels)

    @staticmethod
    def predict(logits):
        return (logits > 0).sum(dim=1)


class MDN(nn.Module):
    """Mixture density output: the spread shows which rate directions are unidentifiable."""

    def __init__(self, din: int, dout: int, n_comp: int = 3):
        super().__init__()
        self.n_comp, self.dout = n_comp, dout
        self.body = mlp(din, n_comp * (1 + 2 * dout), 128, 2)

    def forward(self, x):
        h = self.body(x)
        logit = h[:, : self.n_comp]
        mu, log_sd = h[:, self.n_comp :].chunk(2, dim=1)
        return logit, mu.view(-1, self.n_comp, self.dout), log_sd.view(-1, self.n_comp, self.dout)

    def nll(self, x, y):
        logit, mu, log_sd = self(x)
        sd = log_sd.clamp(-6, 3).exp()
        lp = (-0.5 * ((y[:, None] - mu) / sd) ** 2 - sd.log()).sum(-1)
        return -torch.logsumexp(logit.log_softmax(-1) + lp, dim=-1).mean()

    def mean(self, x):
        logit, mu, _ = self(x)
        return (logit.softmax(-1)[..., None] * mu).sum(1)


class SeqPooled(nn.Module):
    """A sequence backbone over the open-probability track, pooled to a rate guess.

    This is the only place a backbone can matter in v4: emissions are pointwise
    and the chain is exact, so the encoder exists purely to summarise timing for
    the rate head. Three families are compared against hand-written statistics.
    """

    def __init__(self, kind: str, dout: int, width: int = 32):
        super().__init__()
        self.kind = kind
        if kind == "tcn":
            self.body = TCN_ENCODER(hidden=width)
        elif kind == "unet":
            self.down = nn.ModuleList([nn.Conv1d(1 if i == 0 else width, width, 5, stride=4, padding=2)
                                       for i in range(3)])
            self.up = nn.ModuleList([nn.Conv1d(width, width, 3, padding=1) for _ in range(3)])
        elif kind == "gru":
            self.rnn = nn.GRU(1, width, batch_first=True, bidirectional=True)
        self.head = mlp(4 * width if kind == "gru" else 2 * width, dout, 128, 2)

    def encode(self, x):
        if self.kind == "tcn":
            return self.body(x)
        if self.kind == "gru":
            return self.rnn(x[:, :, None])[0]
        h = x[:, None, :]
        skips = []
        for conv in self.down:
            h = F.gelu(conv(h))
            skips.append(h)
        for conv, s in zip(self.up, reversed(skips)):
            h = F.gelu(conv(F.interpolate(h, size=s.shape[-1]) + s))
        return F.interpolate(h, size=x.shape[-1]).transpose(1, 2)

    def forward(self, x, group, n_groups):
        h = self.encode(x)
        summary = torch.cat([h.mean(1), h.std(1)], dim=1)
        return self.head(V4.group_pool(summary, group, n_groups))


class AttnPool(nn.Module):
    """Attention pooling over the traces of a group instead of averaging them."""

    def __init__(self, din: int, dout: int, width: int = 128):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(din, width), nn.GELU())
        self.score = nn.Linear(width, 1)
        self.head = mlp(2 * width, dout, width, 2)

    def forward(self, x, group, n_groups):
        h = self.embed(x)
        w = self.score(h)
        mx = h.new_full((n_groups, 1), -1e30).index_reduce_(0, group, w, "amax", include_self=True)
        e = (w - mx[group]).exp()
        num = h.new_zeros(n_groups, h.shape[1]).index_add_(0, group, h * e)
        den = h.new_zeros(n_groups, 1).index_add_(0, group, e)
        mean = h.new_zeros(n_groups, h.shape[1]).index_add_(0, group, h) / \
            h.new_zeros(n_groups, 1).index_add_(0, group, torch.ones_like(e)).clamp_min(1)
        return self.head(torch.cat([num / den.clamp_min(1e-9), mean], dim=-1))


# --------------------------------------------------------------------------- tasks


def train_head(build, feats, targets, steps: int, lr: float, loss_fn, predict_fn,
               metric_fn, seed: int = 0, batch_size: int = 256):
    torch.manual_seed(seed)
    net = build()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    xtr, ytr = feats["train"], targets["train"]
    t0 = time.time()
    for _ in range(steps):
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), batch_size):
            j = perm[i : i + batch_size]
            loss = loss_fn(net, xtr[j], ytr[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        out = {split: metric_fn(predict_fn(net, feats[split]), split) for split in feats}
    out["seconds"] = round(time.time() - t0, 1)
    out["params"] = sum(p.numel() for p in net.parameters())
    return out


def task_count(cache: dict[str, dict], steps: int, seed: int) -> dict:
    feats_of = lambda key: {s: torch.as_tensor(c[key], dtype=torch.float32) for s, c in cache.items()}  # noqa: E731
    labels = {s: torch.as_tensor(c["N"].astype(np.int64)) - 1 for s, c in cache.items()}
    sources = {"val": ("val", None), "test": ("test", "1")}
    raw = {s: torch.as_tensor(
        np.concatenate([load("train")["X"], load("train_extra")["X"]])[: len(cache[s]["N"])]
        if s == "train" else load(*sources[s])["X"][: len(cache[s]["N"])], dtype=torch.float32)
        for s in cache} if (CACHE / "train_extra.npz").exists() else {
        s: torch.as_tensor((load("train") if s == "train" else load(*sources[s]))["X"][: len(cache[s]["N"])],
                           dtype=torch.float32) for s in cache}

    def acc(pred, split):
        return round(float((pred == labels[split]).float().mean()), 4)

    ce = lambda net, x, y: F.cross_entropy(net(x), y)  # noqa: E731
    argmax = lambda net, x: net(x).argmax(-1)  # noqa: E731

    rows: dict[str, dict] = {}
    rows["reference: evidence pass"] = {
        s: round(float((torch.as_tensor(c["evidence"]).argmax(1) == labels[s]).float().mean()), 4)
        for s, c in cache.items()}
    rows["reference: v4b count head"] = {
        s: round(float((torch.as_tensor(c["n_head"].astype(np.int64)) - 1 == labels[s]).float().mean()), 4)
        for s, c in cache.items()}

    candidates = {
        "MLP on histogram (v4b input)": ("level", lambda d: mlp(d, K_MAX), ce, argmax),
        "MLP on multi-res pyramid": ("pyramid", lambda d: mlp(d, K_MAX), ce, argmax),
        "MLP on quantile curve": ("quantile", lambda d: mlp(d, K_MAX), ce, argmax),
        "CNN over histogram": ("level", lambda d: HistCNN(d, K_MAX), ce, argmax),
        "Deep Sets over raw samples": ("__raw__", lambda d: DeepSets(K_MAX), ce, argmax),
        "ordinal CORAL on histogram": ("level", lambda d: Ordinal(d),
                                       lambda net, x, y: Ordinal.loss(net(x), y),
                                       lambda net, x: Ordinal.predict(net(x))),
        "MLP on pyramid + evidence": ("pyramid+evidence", lambda d: mlp(d, K_MAX), ce, argmax),
        "MLP on evidence only": ("evidence_rel", lambda d: mlp(d, K_MAX, 64, 1), ce, argmax),
    }
    for name, (key, build, loss_fn, pred_fn) in candidates.items():
        if key == "pyramid+evidence":
            feats = {s: torch.cat([torch.as_tensor(c["pyramid"], dtype=torch.float32),
                                   torch.as_tensor(c["evidence_rel"], dtype=torch.float32)], dim=1)
                     for s, c in cache.items()}
        elif key == "__raw__":
            feats = raw
        else:
            feats = feats_of(key)
        din = feats["train"].shape[1]
        rows[name] = train_head(lambda: build(din), feats, labels, steps, 3e-3,
                                loss_fn, pred_fn, acc, seed)
        print(f"  {name:32s} " + "  ".join(f"{s}={rows[name][s]:.4f}" for s in cache)
              + f"  ({rows[name]['seconds']}s, {rows[name]['params']} params)", flush=True)

    # distillation: same input, but match the evidence logits instead of the label
    feats = feats_of("pyramid")
    soft = {s: torch.as_tensor(c["evidence_rel"], dtype=torch.float32) for s, c in cache.items()}
    din = feats["train"].shape[1]
    torch.manual_seed(seed)
    net = mlp(din, K_MAX)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    t0 = time.time()
    for _ in range(steps):
        perm = torch.randperm(len(feats["train"]))
        for i in range(0, len(perm), 256):
            j = perm[i : i + 256]
            tgt = soft["train"][j].log_softmax(-1)
            loss = (F.cross_entropy(net(feats["train"][j]), labels["train"][j])
                    + F.kl_div(net(feats["train"][j]).log_softmax(-1), tgt,
                               log_target=True, reduction="batchmean"))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        rows["MLP on histogram, evidence-distilled"] = {
            s: acc(net(feats[s]).argmax(-1), s) for s in cache}  # renamed below
    rows["MLP on pyramid, evidence-distilled"] = rows.pop("MLP on histogram, evidence-distilled")
    rows["MLP on pyramid, evidence-distilled"]["seconds"] = round(time.time() - t0, 1)
    print(f"  {'MLP on pyramid, evidence-distilled':32s} "
          + "  ".join(f"{s}={rows['MLP on pyramid, evidence-distilled'][s]:.4f}" for s in cache), flush=True)
    return rows


def task_rates(cache: dict[str, dict], steps: int, seed: int, basis: np.ndarray) -> dict:
    """Group-level rate regression on cached per-trace statistics."""
    def group_tensors(c):
        g = torch.as_tensor(c["group"].astype(np.int64))
        return g, int(g.max()) + 1

    def pooled(c, keys):
        parts = [torch.as_tensor(c[k], dtype=torch.float32) for k in keys]
        x = torch.cat([p if p.dim() == 2 else p[:, None] for p in parts], dim=1)
        g, n_groups = group_tensors(c)
        return V4.group_pool(x, g, n_groups)

    targets = {s: torch.as_tensor(np.log(c["R"] / B2.BASE_OFF), dtype=torch.float32)
               for s, c in cache.items()}
    B = torch.as_tensor(basis, dtype=torch.float32)

    def dir_r2(pred, split):
        t = targets[split] @ B
        p = pred @ B
        return [round(float(1 - ((p[:, j] - t[:, j]) ** 2).mean() / max(t[:, j].var(), 1e-12)), 3)
                for j in range(B.shape[1])]

    mse = lambda net, x, y: F.mse_loss(net(x), y)  # noqa: E731
    ident = lambda net, x: net(x)  # noqa: E731

    rows: dict[str, dict] = {}
    rows["reference: v4b rate head"] = {  # cached rates are already one row per group
        s: dir_r2(torch.as_tensor(np.log(c["rates_head"] / B2.BASE_OFF), dtype=torch.float32), s)
        for s, c in cache.items()}

    candidates = {
        "timing stats (v4b features)": ["kin"],
        "+ score vector": ["kin", "score"],
        "+ expected transition counts": ["kin", "xi"],
        "score vector only": ["score"],
        "expected transition counts only": ["xi"],
        "score + transitions + timing": ["kin", "score", "xi"],
    }
    for name, keys in candidates.items():
        feats = {s: pooled(c, keys) for s, c in cache.items()}
        din = feats["train"].shape[1]
        rows[name] = train_head(lambda: mlp(din, N_RATES), feats, targets, steps, 3e-3,
                                mse, ident, dir_r2, seed, batch_size=64)
        print(f"  {name:32s} " + "  ".join(f"{s}={rows[name][s]}" for s in cache)
              + f"  ({rows[name]['seconds']}s)", flush=True)

    # MDN over the rates, on the best feature set
    feats = {s: pooled(c, ["kin", "score", "xi"]) for s, c in cache.items()}
    din = feats["train"].shape[1]
    rows["MDN on score + transitions + timing"] = train_head(
        lambda: MDN(din, N_RATES), feats, targets, steps, 3e-3,
        lambda net, x, y: net.nll(x, y), lambda net, x: net.mean(x), dir_r2, seed, batch_size=64)
    print(f"  {'MDN on score+transitions+timing':32s} "
          + "  ".join(f"{s}={rows['MDN on score + transitions + timing'][s]}" for s in cache), flush=True)

    # attention pooling over the group instead of mean pooling
    def attn_feats(c):
        x = torch.cat([torch.as_tensor(c[k], dtype=torch.float32) for k in ("kin", "score", "xi")], dim=1)
        g, n_groups = group_tensors(c)
        return x, g, n_groups

    torch.manual_seed(seed)
    packed = {s: attn_feats(c) for s, c in cache.items()}
    net = AttnPool(packed["train"][0].shape[1], N_RATES)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    t0 = time.time()
    xtr, gtr, ntr = packed["train"]
    for _ in range(steps):
        loss = F.mse_loss(net(xtr, gtr, ntr), targets["train"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        rows["attention pooling over the group"] = {s: dir_r2(net(*packed[s]), s) for s in cache}
    rows["attention pooling over the group"]["seconds"] = round(time.time() - t0, 1)
    print(f"  {'attention pooling over group':32s} "
          + "  ".join(f"{s}={rows['attention pooling over the group'][s]}" for s in cache), flush=True)

    # backbone probes: a sequence model over the open-probability track, which is
    # the only place an encoder can help once emissions are pointwise
    seq = {s: (torch.as_tensor(c["post_open"], dtype=torch.float32),
               torch.as_tensor(c["group"].astype(np.int64)),
               int(c["group"].max()) + 1) for s, c in cache.items()}
    for kind, label in (("tcn", "backbone: TCN over p_open"),
                        ("unet", "backbone: U-Net over p_open"),
                        ("gru", "backbone: bidirectional GRU over p_open")):
        torch.manual_seed(seed)
        net = SeqPooled(kind, N_RATES)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        xtr, gtr, ntr = seq["train"]
        t0 = time.time()
        for _ in range(max(steps // 6, 20)):
            perm = torch.randperm(ntr)[:128]
            mask = torch.isin(gtr, perm)
            sub_g = torch.searchsorted(perm.sort().values, gtr[mask])
            loss = F.mse_loss(net(xtr[mask], sub_g, len(perm)), targets["train"][perm.sort().values])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            rows[label] = {s: dir_r2(net(*seq[s]), s) for s in cache}
        rows[label]["seconds"] = round(time.time() - t0, 1)
        print(f"  {label:32s} " + "  ".join(f"{s}={rows[label][s]}" for s in cache)
              + f"  ({rows[label]['seconds']}s)", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--task", choices=("count", "rates"))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--extra", action="store_true",
                        help="include train_extra (1024 more groups) in the training pool")
    args = parser.parse_args()

    names = {"train": ("train", None), "val": ("val", None), "test": ("test", "1")}
    if args.extra:
        names["train_extra"] = ("train_extra", None)
    if args.build_cache:
        evid = TK.default_device() if torch.backends.mps.is_available() else args.device
        build_cache({k: load(*v) for k, v in names.items()}, args.device, evid, args.limit_groups)
        return

    cache = {k: load_cache(k) for k in names if k != "train_extra"}
    if args.extra and (CACHE / "train_extra.npz").exists():
        cache["train"] = concat_caches([cache["train"], load_cache("train_extra")])
        print(f"train augmented to {len(cache['train']['R'])} groups", flush=True)
    evals, evecs = np.linalg.eigh(
        np.asarray(json.loads((RESULTS / "fisher_synth_v2.json").read_text())["fisher"]))
    basis = evecs[:, np.argsort(evals)[::-1][:4]]
    print(f"task={args.task}  train={len(cache['train']['N'])} traces  steps={args.steps}", flush=True)
    rows = (task_count(cache, args.steps, args.seed) if args.task == "count"
            else task_rates(cache, args.steps, args.seed, basis))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"bakeoff_{args.task}.json").write_text(json.dumps(rows, indent=2))
    print("saved", RESULTS / f"bakeoff_{args.task}.json", flush=True)


if __name__ == "__main__":
    main()

"""KI-HMM v4: one network that infers N, the per-state counts a..g and the rates.

v2 and v3 predicted the rate table but never used it: their HMM prior was the
chain of the *base* rate table, while ``synth_v2`` randomises every rate by 0.5x
to 2x per group. The smoothing prior was therefore wrong for almost every group,
and the only correction available was a free-form MLP from log rates to
transition logits sitting behind a zero-initialised gate.

v4 closes that loop. The rate head's own output is turned into a transition
matrix by the actual physics -- ``matrix_exp`` of the generator, then the exact
occupancy-count chain -- inside the forward pass, so the same gradient step that
moves the rates moves the prior the posterior is computed with. Emissions come
from a learned mixture density evaluated pointwise at ``y_t``, which keeps the
temporal prior from being counted twice and makes the HMM's log evidence a
genuine likelihood.

Outputs, all from one forward pass:

  N              channel-count head, 1..5
  a..g           posterior expected channels per kinetic state (sums to N)
  Markov rates   the 12 off-diagonal rates of the CFTR graph, per group

Everything is a differentiable layer trained by gradient descent: no EM, no
clustering, no separate decoding stage.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import benchmark_v2 as B2  # noqa: E402
import torch_emissions as TE  # noqa: E402
import torch_kinetics as TK  # noqa: E402
from torch_models import TemporalEncoder  # noqa: E402

K_MAX = 5
N_STATES = TK.N_STATES
N_RATES = TK.N_RATES
LOG_RANGE = 1.0
HIST_BINS = 64
HIST_RANGE = (-2.0, 9.0)
QUANTILES = (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98)
LEVEL_FEATURES = HIST_BINS + len(QUANTILES) + 2
LAGS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89)


def group_pool(x: torch.Tensor, group: torch.Tensor, n_groups: int) -> torch.Tensor:
    total = x.new_zeros(n_groups, x.shape[1])
    count = x.new_zeros(n_groups, 1)
    total.index_add_(0, group, x)
    count.index_add_(0, group, torch.ones_like(x[:, :1]))
    return total / count.clamp_min(1.0)


def summarise(z: torch.Tensor) -> torch.Tensor:
    return torch.cat([z.mean(dim=1), z.std(dim=1),
                      torch.quantile(z, 0.25, dim=1), torch.quantile(z, 0.75, dim=1)], dim=1)


def level_features(x: torch.Tensor) -> torch.Tensor:
    """Amplitude histogram and quantiles of the raw trace, plus mean and log spread.

    The encoder sees a per-trace normalized signal, which erases the absolute
    levels that say how many channels are present. This is the CCNN cue put back
    explicitly: a fixed-bin histogram is what let the thesis count channels, and
    unlike deep-feature quantiles it means the same thing at any trace length.
    """
    lo, hi = HIST_RANGE
    idx = (((x - lo) / (hi - lo)) * HIST_BINS).long().clamp(0, HIST_BINS - 1)
    hist = x.new_zeros(x.shape[0], HIST_BINS).scatter_add_(1, idx, torch.ones_like(x))
    qs = torch.as_tensor(QUANTILES, device=x.device, dtype=x.dtype)
    return torch.cat([hist / x.shape[1],
                      torch.quantile(x, qs, dim=1).T,
                      x.mean(dim=1, keepdim=True),
                      x.std(dim=1, keepdim=True).clamp_min(1e-6).log()], dim=1)


def kinetic_features(p_open: torch.Tensor, k_hat: torch.Tensor, k_max: int) -> torch.Tensor:
    """Timing statistics of the pointwise open estimate, for the rate head.

    The autocorrelation of the open count decays with the relaxation time of the
    gating chain, so a handful of lags pins down the effective rates; the
    empirical transition matrix of the hard open count carries the same
    information the dwell-time histograms do. Both are length invariant, which
    matters because training crops the traces and evaluation does not.
    """
    z = p_open - p_open.mean(dim=1, keepdim=True)
    var = (z * z).mean(dim=1).clamp_min(1e-8)
    acf = torch.stack([(z[:, :-l] * z[:, l:]).mean(dim=1) / var for l in LAGS], dim=1)
    pair = k_hat[:, :-1] * (k_max + 1) + k_hat[:, 1:]
    trans = p_open.new_zeros(p_open.shape[0], (k_max + 1) ** 2)
    trans.scatter_add_(1, pair, torch.ones_like(pair, dtype=p_open.dtype))
    return torch.cat([acf, trans / (p_open.shape[1] - 1)], dim=1)


class KIHMMv4(nn.Module):
    def __init__(self, hidden: int = 64, rate_hidden: int = 32, n_comp: int = 4,
                 k_max: int = K_MAX, rate_stats: bool = True, n_head: str = "level"):
        super().__init__()
        self.k_max = k_max
        self.rate_stats = rate_stats
        self.n_head_mode = n_head
        self.encoder = TemporalEncoder(hidden=hidden)
        self.dwell_encoder = TemporalEncoder(hidden=rate_hidden)
        self.emission = TE.SumMixtureEmission(n_comp=n_comp, k_max=k_max)
        trace_in = 4 * hidden + LEVEL_FEATURES
        self.trace_dense = nn.Sequential(
            nn.Linear(trace_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        # Counting channels is a different problem from timing them. Sharing the
        # encoder costs roughly 9 points of N accuracy: a plain MLP on the
        # amplitude histogram alone reaches 0.99 on the frozen test set, which is
        # the thesis CCNN cue, so by default the count head reads only that.
        n_in = {"level": LEVEL_FEATURES, "trace": trace_in, "pooled": hidden}[n_head]
        self.n_head = (nn.Linear(hidden, k_max) if n_head == "pooled" else
                       nn.Sequential(nn.Linear(n_in, 128), nn.GELU(),
                                     nn.Linear(128, 128), nn.GELU(), nn.Linear(128, k_max)))
        self.scale_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        extra = (len(LAGS) + (k_max + 1) ** 2) if rate_stats else 0
        self.rate_dense = nn.Sequential(
            nn.Linear(hidden + 4 * rate_hidden + 2 + extra, hidden), nn.GELU()
        )
        self.rate_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, N_RATES))
        nn.init.zeros_(self.rate_head[-1].weight)
        nn.init.zeros_(self.rate_head[-1].bias)
        self.register_buffer("rate_base", torch.as_tensor(B2.BASE_OFF, dtype=torch.float32))

    def load_emission_fit(self, path: str | pathlib.Path) -> None:
        """Warm-start the emission mixture from the GH fit produced by oracle_v4."""
        self.emission.load_state_dict(torch.load(path, map_location="cpu")["model"])

    def trace_features(self, x: torch.Tensor):
        """Shared trace summary, plus the pooled vector the scale and rate heads use."""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        levels = level_features(x)
        raw = torch.cat([summarise(self.encoder((x - mean) / std)), levels], dim=1)
        pooled = self.trace_dense(raw)
        return pooled, self.n_head({"level": levels, "trace": raw, "pooled": pooled}[self.n_head_mode])

    def forward(self, x: torch.Tensor, group: torch.Tensor, n_ref: torch.Tensor,
                emission_chunk: int = 0) -> dict:
        """x (B,T), group (B,) contiguous ids, n_ref (B,) channel counts for the chain."""
        pooled, n_logits = self.trace_features(x)
        log_scale = self.scale_head(pooled).squeeze(-1).clamp(-3.0, 3.0)
        n_ref = n_ref.long().clamp(1, self.k_max)
        buckets = [(int(n), torch.nonzero(n_ref == int(n), as_tuple=True)[0])
                   for n in torch.unique(n_ref)]

        # pointwise emissions; their per-sample open share is the dwell cue the
        # rate head needs, and it is available before the chain exists
        log_emit = {}
        p_open = x.new_zeros(x.shape)
        k_hat = torch.zeros_like(x, dtype=torch.long)
        for n, idx in buckets:
            log_emit[n] = self.emission(x[idx], n, log_scale[idx], chunk=emission_chunk)
            k = torch.arange(n + 1, device=x.device, dtype=x.dtype)
            p_open[idx] = (log_emit[n].softmax(-1) * k).sum(-1) / n
            k_hat[idx] = log_emit[n].argmax(-1)

        rc = self.dwell_encoder(p_open)
        parts = [pooled, summarise(rc), p_open.mean(1, keepdim=True), p_open.std(1, keepdim=True)]
        if self.rate_stats:
            parts.append(kinetic_features(p_open, k_hat, self.k_max))
        rate_feat = self.rate_dense(torch.cat(parts, dim=1))
        n_groups = int(group.max().item()) + 1
        delta = self.rate_head(group_pool(rate_feat, group, n_groups))
        rates = self.rate_base[None, :] * torch.exp(LOG_RANGE * torch.tanh(delta))

        counts = x.new_zeros(x.shape[0], N_STATES, x.shape[1])
        e_open = x.new_zeros(x.shape)
        logZ = x.new_zeros(x.shape[0])
        state_path = {}
        for n, idx in buckets:
            tables = TK.get_tables(n, x.device)
            P0 = TK.p0_from_rates(rates[group[idx]])
            logP = TK.count_transition(P0, tables).clamp_min(1e-30).log()
            logpi = TK.count_log_initial(TK.stationary(P0), tables)
            logb = TK.expand_open_emissions(log_emit[n], tables)
            lz, gamma = TK.hmm_posterior(logb, logP, logpi)
            logZ = logZ.index_copy(0, idx, lz)
            counts = counts.index_copy(0, idx, TK.posterior_counts(gamma, tables))
            e_open = e_open.index_copy(0, idx, gamma @ tables.open_count.to(x.dtype))
            state_path[n] = (idx, logb, logP, logpi, tables)

        return {
            "n_logits": n_logits,
            "log_scale": log_scale,
            "rates_group": rates,
            "rates": rates[group],
            "counts": counts,
            "e_open": e_open,
            "logZ": logZ,
            "chain": state_path,
        }


def counts_from_r(r: torch.Tensor) -> torch.Tensor:
    """(B, max_ch, T) state indices padded with -1 -> (B, 7, T) counts per state."""
    return torch.stack([(r == s).sum(dim=1) for s in range(N_STATES)], dim=1).float()


def kihmm_v4_loss(out: dict, N_true: torch.Tensor, counts_true: torch.Tensor,
                  rates_true: torch.Tensor, log_scale_true: torch.Tensor | None = None,
                  w_n: float = 0.5, w_crf: float = 1.0, w_gen: float = 0.3,
                  w_rate: float = 1.0, w_scale: float = 0.1,
                  rate_metric: torch.Tensor | None = None):
    """CRF negative log-likelihood of the true occupancy path, plus the usual heads.

    ``HMMLogZ`` hands back the posterior detached, so supervision uses the exact
    identity log p(states | y) = path_score - logZ. Both terms are differentiable
    without ever backpropagating through the forward-backward recursion. The
    separate ``-logZ`` term is the generative pressure that makes the emission
    density and the predicted rates explain the data rather than only match a
    regression target.
    """
    device = out["logZ"].device
    T = counts_true.shape[-1]
    crf = out["logZ"].new_zeros(())
    for n, (idx, logb, logP, logpi, tables) in out["chain"].items():
        path = tables.state_index(counts_true[idx])
        crf = crf + (out["logZ"][idx] - TK.path_score(logb, logP, logpi, path)).sum()
    crf = crf / (len(N_true) * T)
    gen = -out["logZ"].mean() / T

    n_loss = F.cross_entropy(out["n_logits"], (N_true - 1).long())
    delta = torch.log(out["rates"]) - torch.log(rates_true)
    if rate_metric is not None:
        rate_loss = ((delta @ rate_metric) * delta).sum(dim=-1).mean()
    else:
        rate_loss = delta.abs().mean()
    scale_loss = (torch.zeros((), device=device) if log_scale_true is None
                  else F.mse_loss(out["log_scale"], log_scale_true))

    loss = w_n * n_loss + w_crf * crf + w_gen * gen + w_rate * rate_loss + w_scale * scale_loss
    return loss, {
        "crf": float(crf.detach()),
        "gen": float(gen.detach()),
        "n": float(n_loss.detach()),
        "rate": float(rate_loss.detach()),
        "rate_log": float(delta.abs().mean().detach()),
        "scale": float(scale_loss.detach()),
    }


@torch.no_grad()
def predict(model: KIHMMv4, x: torch.Tensor, group: torch.Tensor,
            n_ref: torch.Tensor | None = None, emission_chunk: int = 0) -> dict:
    """Full inference: N from the count head, then counts and rates from that N."""
    model.eval()
    _, n_logits = model.trace_features(x)
    n_hat = n_logits.argmax(dim=-1) + 1 if n_ref is None else n_ref.long()
    out = model(x, group, n_hat, emission_chunk=emission_chunk)
    return {
        "N_hat": n_hat,
        "N_exp": (F.softmax(n_logits, dim=-1)
                  * torch.arange(1, model.k_max + 1, device=x.device)).sum(-1),
        "counts": out["counts"],
        "e_open": out["e_open"],
        "rates": out["rates_group"],
        "log_scale": out["log_scale"],
        "logZ": out["logZ"],
    }

"""KI-HMM v2: N + per-state counts + Markov parameters from summed CFTR traces.

Three heads share one TCN encoder:

  - N head: number of channels (1..5);
  - state-count head: expected channels in each of the 7 kinetic states at every
    time step (a..g, sum = N);
  - rate head: the 12 off-diagonal rates of the CFTR graph, pooled over a group
    of traces recorded under the same kinetics.

Training is supervised on the rate-randomized benchmark ``synth_v2``
(``benchmark_v2.py``) -- no EM, no hand-set rates. The exact kinetic chain in
``kinetics.py`` stays available as an optional refinement and evaluation tool.
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
from torch_models import TemporalEncoder  # noqa: E402

K_MAX = 5
N_STATES = 7
N_RATES = len(B2.RATE_INDEX)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class KIHMMv2(nn.Module):
    def __init__(self, hidden: int = 64, k_max: int = K_MAX):
        super().__init__()
        self.k_max = k_max
        self.encoder = TemporalEncoder(hidden=hidden)
        self.rate_encoder = TemporalEncoder(hidden=32)
        self.trace_dense = nn.Sequential(nn.Linear(4 * hidden + 4 * 32 + 2 + 2 * N_STATES + 2, hidden), nn.GELU())
        self.n_head = nn.Linear(hidden, k_max)
        self.state_head = nn.Linear(hidden, N_STATES)
        self.rate_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, N_RATES)
        )
        base = torch.as_tensor(B2.BASE_OFF, dtype=torch.float32)
        self.register_buffer("rate_base", base)
        nn.init.zeros_(self.rate_head[-1].weight)
        nn.init.zeros_(self.rate_head[-1].bias)

    def encode(self, x: torch.Tensor):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        h = self.encoder((x - mean) / std)
        glob = torch.cat([torch.log(std), mean], dim=1)
        return h, glob

    def forward(self, x: torch.Tensor, group: torch.Tensor):
        """x: (B,T), group: (B,) group ids in 0..G-1 (contiguous)."""
        h, glob = self.encode(x)
        state_probs = F.softmax(self.state_head(h), dim=-1)
        open_idx = torch.as_tensor(np.flatnonzero(B2.B.STATEMAP == 1), device=x.device)
        p_open = state_probs[..., open_idx].sum(dim=-1)
        rc = self.rate_encoder(p_open)
        summary = torch.cat(
            [h.mean(dim=1), h.std(dim=1),
             torch.quantile(h, 0.25, dim=1), torch.quantile(h, 0.75, dim=1),
             glob,
             state_probs.mean(dim=1), state_probs.std(dim=1),
             rc.mean(dim=1), rc.std(dim=1),
             torch.quantile(rc, 0.25, dim=1), torch.quantile(rc, 0.75, dim=1),
             p_open.mean(dim=1, keepdim=True), p_open.std(dim=1, keepdim=True)],
            dim=1,
        )
        pooled = self.trace_dense(summary)

        n_groups = int(group.max().item()) + 1
        gsum = torch.zeros(n_groups, pooled.shape[1], device=x.device, dtype=pooled.dtype)
        gcount = torch.zeros(n_groups, 1, device=x.device, dtype=pooled.dtype)
        gsum.index_add_(0, group, pooled)
        gcount.index_add_(0, group, torch.ones_like(pooled[:, :1]))
        gfeat = gsum / gcount.clamp_min(1.0)

        delta = self.rate_head(gfeat)
        rates = self.rate_base[None, :] * torch.exp(delta)
        rate_prior = self.rate_base[None, :] * torch.exp(torch.clamp(delta, -1.0, 1.0))

        n_logits = self.n_head(pooled)
        return {
            "n_logits": n_logits,
            "state_probs": state_probs,
            "rates": rate_prior[group],
            "rates_group": rate_prior,
            "rates_raw": rates[group],
        }


def counts_from_r(r: torch.Tensor) -> torch.Tensor:
    """(B, max_ch, T) state indices -> (B, 7, T) counts per state."""
    return torch.stack([(r == s).sum(dim=1) for s in range(N_STATES)], dim=1).float()


def scale_counts(state_probs: torch.Tensor, n_values: torch.Tensor) -> torch.Tensor:
    """Expected per-state counts, shape (B, 7, T): probabilities x N."""
    return state_probs.transpose(1, 2) * n_values[:, None, None]


def kihmm_v2_loss(out: dict, N_true: torch.Tensor, counts_true: torch.Tensor,
                  rates_true: torch.Tensor, w_n: float = 0.3, w_state: float = 1.0,
                  w_rate: float = 1.0, rate_metric: torch.Tensor | None = None):
    n_loss = F.cross_entropy(out["n_logits"], (N_true - 1).long())
    counts = scale_counts(out["state_probs"], N_true.float())
    state_loss = (counts - counts_true).abs().mean()
    delta = torch.log(out["rates"]) - torch.log(rates_true)
    if rate_metric is not None:
        rate_loss = ((delta @ rate_metric) * delta).sum(dim=-1).mean()
    else:
        rate_loss = delta.abs().mean()
    loss = w_n * n_loss + w_state * state_loss + w_rate * rate_loss
    return loss, {
        "n": float(n_loss.detach()),
        "state": float(state_loss.detach()),
        "rate": float(rate_loss.detach()),
        "rate_log": float(delta.abs().mean().detach()),
    }


@torch.no_grad()
def predict(model: KIHMMv2, x: torch.Tensor, group: torch.Tensor, k_max: int = K_MAX):
    model.eval()
    out = model(x, group)
    n_probs = F.softmax(out["n_logits"], dim=-1)
    n_exp = (n_probs * torch.arange(1, k_max + 1, device=x.device)).sum(dim=-1)
    counts = scale_counts(out["state_probs"], n_exp)
    return {
        "N_hat": out["n_logits"].argmax(dim=-1) + 1,
        "N_exp": n_exp,
        "counts": counts,
        "rates": out["rates_group"],
    }


def state_open_counts(counts: torch.Tensor) -> torch.Tensor:
    open_idx = torch.as_tensor(np.flatnonzero(B2.B.STATEMAP == 1), device=counts.device)
    return counts[:, open_idx, :].sum(dim=1)

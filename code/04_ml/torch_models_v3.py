"""KI-HMM v3: neural HMM head for open-count marginals (full neural model).

Same TCN encoder as v2, but the count output comes from a linear-chain HMM over
the open count k = 0..N:

  - emissions: encoder features -> logits over k (masked to k <= N);
  - transitions: initialized from the biophysical count chain of the base rate
    table, then corrected per trace and per group by small neural nets that see
    the encoder summary and the rate head's predicted rates;
  - forward-backward returns per-timestep marginals, so dwell structure is
    respected by construction.

Per-state counts a..g are then assembled from the open-count marginal and a
7-state composition head (open/closed shares), so they still sum to N.
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
import torch_models as TM  # noqa: E402

K_MAX = 5
N_STATES = 7
N_RATES = len(B2.RATE_INDEX)
MASK_NEG = -1e6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def base_chain_buffers(k_max: int = K_MAX):
    logP = torch.full((k_max, k_max + 1, k_max + 1), MASK_NEG)
    logpi = torch.full((k_max, k_max + 1), MASK_NEG)
    for n in range(1, k_max + 1):
        Pk, pik = TM.open_count_chain(n)
        logP[n - 1, : n + 1, : n + 1] = torch.log(torch.as_tensor(Pk, dtype=torch.float32))
        logpi[n - 1, : n + 1] = torch.log(torch.as_tensor(pik, dtype=torch.float32))
    return logP, logpi


class KIHMMv3(nn.Module):
    def __init__(self, hidden: int = 64, k_max: int = K_MAX):
        super().__init__()
        self.k_max = k_max
        self.encoder = TM.TemporalEncoder(hidden=hidden)
        self.rate_encoder = TM.TemporalEncoder(hidden=32)
        self.trace_dense = nn.Sequential(
            nn.Linear(4 * hidden + 4 * 32 + 2 + 2 * N_STATES + 2, hidden), nn.GELU()
        )
        self.n_head = nn.Linear(hidden, k_max)
        self.state_head = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.GELU(), nn.Linear(hidden, N_STATES)
        )
        self.emit_head = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.GELU(), nn.Linear(hidden, k_max + 1)
        )
        self.rate_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, N_RATES))
        self.trans_delta = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, (k_max + 1) ** 2)
        )
        self.rate_to_trans = nn.Sequential(
            nn.Linear(N_RATES, hidden), nn.GELU(), nn.Linear(hidden, (k_max + 1) ** 2)
        )
        for layer in (self.trans_delta[-1], self.rate_to_trans[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        logP, logpi = base_chain_buffers(k_max)
        self.register_buffer("logP_base", logP)
        self.register_buffer("logpi_base", logpi)
        self.register_buffer("rate_base", torch.as_tensor(B2.BASE_OFF, dtype=torch.float32))
        self.register_buffer("k_index", torch.arange(k_max + 1, dtype=torch.float32))
        self.trans_gate = nn.Parameter(torch.zeros(1))

    def encode(self, x: torch.Tensor):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        h = self.encoder((x - mean) / std)
        glob = torch.cat([torch.log(std), mean], dim=1)
        return h, glob

    def forward(self, x: torch.Tensor, group: torch.Tensor, n_ref: torch.Tensor | None = None):
        h, glob = self.encode(x)
        z = torch.cat([h, x[..., None]], dim=-1)
        state_probs = F.softmax(self.state_head(z), dim=-1)
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
        rates = self.rate_base[None, :] * torch.exp(torch.clamp(delta, -1.0, 1.0))
        n_logits = self.n_head(pooled)
        if n_ref is None:
            n_ref = n_logits.argmax(dim=-1) + 1
        n_ref = n_ref.long().clamp(1, self.k_max)

        # emissions over open counts, masked to k <= N
        emit = self.emit_head(z)
        k_idx = torch.arange(self.k_max + 1, device=x.device)[None, None, :]
        emit = emit.masked_fill(k_idx > n_ref[:, None, None], MASK_NEG)
        # transitions: base chain + per-trace + per-group (rate) corrections
        base = self.logP_base[n_ref - 1]
        corr = self.trans_delta(pooled).reshape(-1, self.k_max + 1, self.k_max + 1)
        rc_trans = self.rate_to_trans(torch.log(rates)).reshape(-1, self.k_max + 1, self.k_max + 1)
        per_group = rc_trans[group]
        logP = base + self.trans_gate * (corr + per_group)
        bad = (k_idx > n_ref[:, None, None]) | (k_idx.transpose(1, 2) > n_ref[:, None, None])
        logP = logP.masked_fill(bad, MASK_NEG)
        logP = F.log_softmax(logP, dim=-1)
        logpi = self.logpi_base[n_ref - 1]

        log_gamma, logZ = TM.batched_hmm_posterior(emit, logP, logpi)
        gamma = log_gamma.exp()
        e_open = (gamma * self.k_index).sum(dim=-1)  # (B,T)

        # assemble per-state counts from marginal open mass and composition head
        open_share = state_probs[..., open_idx].sum(dim=-1, keepdim=True)
        closed_idx = torch.as_tensor(
            [i for i in range(N_STATES) if i not in open_idx.tolist()], device=x.device
        )
        closed_share = state_probs[..., closed_idx].sum(dim=-1, keepdim=True)
        open_mass = (e_open / n_ref.float()[:, None])[:, :, None]
        counts_open = state_probs[..., open_idx] / open_share.clamp_min(1e-6) * open_mass * n_ref[:, None, None]
        counts_closed = (state_probs[..., closed_idx] / closed_share.clamp_min(1e-6)
                         * (1.0 - open_mass) * n_ref[:, None, None])
        counts = torch.empty_like(state_probs)
        counts[..., open_idx] = counts_open
        counts[..., closed_idx] = counts_closed
        counts = counts.transpose(1, 2)  # (B,7,T)

        return {
            "n_logits": n_logits,
            "emit_logits": emit,
            "state_probs": state_probs,
            "log_gamma": log_gamma,
            "gamma": gamma,
            "e_open": e_open,
            "counts": counts,
            "rates": rates[group],
            "rates_group": rates,
            "logZ": logZ,
        }


def kihmm_v3_loss(out, N_true, open_true, counts_true, rates_true,
                  w_n: float = 0.5, w_nll: float = 1.0, w_emit: float = 0.3,
                  w_state: float = 0.5, w_rate: float = 1.0,
                  rate_metric: torch.Tensor | None = None):
    n_loss = F.cross_entropy(out["n_logits"], (N_true - 1).long())
    tgt = open_true.long()
    nll = -out["log_gamma"].gather(2, tgt[:, :, None]).squeeze(-1).mean()
    emit_ce = F.cross_entropy(out["emit_logits"].reshape(-1, out["emit_logits"].shape[-1]),
                              tgt.reshape(-1))
    state_loss = (out["counts"] - counts_true).abs().mean()
    delta = torch.log(out["rates"]) - torch.log(rates_true)
    if rate_metric is not None:
        rate_loss = ((delta @ rate_metric) * delta).sum(dim=-1).mean()
    else:
        rate_loss = delta.abs().mean()
    loss = (w_n * n_loss + w_nll * nll + w_emit * emit_ce
            + w_state * state_loss + w_rate * rate_loss)
    return loss, {
        "n": float(n_loss.detach()),
        "nll": float(nll.detach()),
        "emit": float(emit_ce.detach()),
        "state": float(state_loss.detach()),
        "rate": float(rate_loss.detach()),
    }


@torch.no_grad()
def predict_v3(model: KIHMMv3, x: torch.Tensor, group: torch.Tensor):
    model.eval()
    out = model(x, group, n_ref=None)
    n_exp = (F.softmax(out["n_logits"], dim=-1)
             * torch.arange(1, model.k_max + 1, device=x.device)).sum(-1)
    return {
        "N_hat": out["n_logits"].argmax(dim=-1) + 1,
        "N_exp": n_exp,
        "counts": out["counts"],
        "e_open": out["e_open"],
        "rates": out["rates_group"],
        "gamma": out["gamma"],
    }

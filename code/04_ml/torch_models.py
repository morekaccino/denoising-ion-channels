"""KI-HMM: kinetics-informed neural hidden Markov model for multi-channel CFTR counting.

The thesis treated factorial HMMs as intractable for multi-channel recordings and
used a windowed LSTM instead. KI-HMM keeps the exact biophysical transition
structure (the open-count chain derived from the 7-state CFTR kinetics) but lets a
neural encoder learn the emission model from data, and infers the number of
channels N trans-dimensionally:

    X -> TCN encoder -> per-sample emission scores p(k_t | x) --.
                                                                +--> exact forward
    N candidates -> kinetic chain P_N, pi_N -------------------'    algorithm
                                                                    |
                                             p(open count | X, N), log evidence_N

Training maximises the smoothed posterior likelihood (structured loss) plus
per-sample and channel-count auxiliary losses. Inference runs every candidate N
through the same N-conditioned network and selects N by the learned count head or
by marginal likelihood.

This combination — exact kinetic chain + learned emissions + trans-dimensional N
inference — has not been applied to multi-channel patch-clamp analysis.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import benchmark as B  # noqa: E402
import kinetics as K  # noqa: E402

K_MAX = 5
MASK_NEG = -1e6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def open_count_chain(N: int, P0: np.ndarray | None = None):
    """Stationary-weighted lumping of the exact count chain onto the open count."""
    P0 = B.p0_matrix() if P0 is None else P0
    states = K.count_states(N)
    P = K.count_transition_matrix(N, P0)
    pi = K.count_stationary(N, P0)
    k = states[:, B.STATEMAP == 1].sum(axis=1).astype(int)
    pi_k = np.zeros(N + 1)
    np.add.at(pi_k, k, pi)
    Pk = np.zeros((N + 1, N + 1))
    for s in range(len(states)):
        if pi[s] <= 0:
            continue
        for kk in range(N + 1):
            Pk[k[s], kk] += pi[s] * P[s][k == kk].sum()
    Pk /= np.clip(Pk.sum(axis=1, keepdims=True), 1e-300, None)
    Pk = np.clip(Pk, 1e-12, None)
    Pk /= Pk.sum(axis=1, keepdims=True)
    return Pk.astype(np.float32), (pi_k / max(pi_k.sum(), 1e-300)).astype(np.float32)


_CHAIN_CACHE: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def chain_tensors(N: int, device: str | None = None):
    device = device or DEVICE
    key = (N, str(device))
    if key not in _CHAIN_CACHE:
        Pk, pik = open_count_chain(N)
        _CHAIN_CACHE[key] = (
            torch.log(torch.as_tensor(Pk, device=device)),
            torch.log(torch.as_tensor(pik, device=device)),
        )
    return _CHAIN_CACHE[key]


def batched_chain_tensors(n_channels: torch.Tensor, k_max: int = K_MAX, device: str | None = None):
    """Per-trace kinetic chains padded to k_max+1 states (invalid states are -inf)."""
    device = device or str(n_channels.device)
    Bn = n_channels.shape[0]
    S = k_max + 1
    logP = torch.full((Bn, S, S), MASK_NEG, device=device)
    logpi = torch.full((Bn, S), MASK_NEG, device=device)
    for n in torch.unique(n_channels).tolist():
        Pn, pin = chain_tensors(int(n), device)
        idx = n_channels == int(n)
        logP[idx, : n + 1, : n + 1] = Pn
        logpi[idx, : n + 1] = pin
    return logP, logpi


def batched_hmm_posterior(logB: torch.Tensor, logP: torch.Tensor, logpi: torch.Tensor):
    """Batched log-space forward-backward. logB: (B,T,S), logP: (B,S,S), logpi: (B,S).

    Returns log-smoothed marginals log_gamma (B,T,S) and log evidence (B,).
    """
    Bn, T, S = logB.shape
    logalpha = torch.empty_like(logB)
    logalpha[:, 0] = logpi + logB[:, 0]
    for t in range(1, T):
        logalpha[:, t] = torch.logsumexp(logalpha[:, t - 1][:, :, None] + logP, dim=1) + logB[:, t]
    logbeta = torch.zeros_like(logB)
    for t in range(T - 2, -1, -1):
        logbeta[:, t] = torch.logsumexp(logP + logB[:, t + 1][:, None, :] + logbeta[:, t + 1][:, None, :], dim=-1)
    log_gamma = torch.log_softmax(logalpha + logbeta, dim=-1)
    logZ = torch.logsumexp(logalpha[:, -1], dim=-1)
    return log_gamma, logZ


def kihmm_forward(model: KIHMM, x: torch.Tensor, n_channels: torch.Tensor):
    """Emission scores from the network + exact kinetic-chain smoothing.

    Returns log_gamma (B,T,K_MAX+1), log evidence (B,), raw logits, count logits.
    """
    logits, count_logits = model(x, n_channels)
    logB = F.log_softmax(logits, dim=-1)
    logP, logpi = batched_chain_tensors(n_channels, device=str(x.device))
    log_gamma, logZ = batched_hmm_posterior(logB, logP, logpi)
    return log_gamma, logZ, logits, count_logits


class TemporalEncoder(nn.Module):
    def __init__(self, hidden: int = 64, dilations=(1, 2, 4, 8, 16, 32), kernel: int = 5):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_ch = 1
        for d in dilations:
            pad = d * (kernel - 1) // 2
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, hidden, kernel, padding=pad, dilation=d),
                    nn.GELU(),
                    nn.Conv1d(hidden, hidden, 1),
                )
            )
            in_ch = hidden
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x[:, None, :]
        for block in self.blocks:
            h = h + block(h)
        return self.norm(h.transpose(1, 2))


class KIHMM(nn.Module):
    def __init__(self, k_max: int = K_MAX, hidden: int = 64, emb: int = 16):
        super().__init__()
        self.k_max = k_max
        self.encoder = TemporalEncoder(hidden=hidden)
        self.n_embed = nn.Embedding(k_max + 1, emb)
        self.film = nn.Linear(emb + 2, 2 * hidden)
        self.emit_head = nn.Linear(hidden, k_max + 1)
        self.count_head = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.GELU(), nn.Linear(hidden, k_max + 1))

    def features(self, x: torch.Tensor, n_channels: torch.Tensor):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        xn = (x - mean) / std
        h0 = self.encoder(xn)
        glob = torch.cat([torch.log(std), mean], dim=1)
        cond = torch.cat([self.n_embed(n_channels - 1), glob], dim=1)
        scale, shift = self.film(cond).chunk(2, dim=1)
        h = h0 * (1 + scale[:, None, :]) + shift[:, None, :]
        return h, h0, glob

    def forward(self, x: torch.Tensor, n_channels: torch.Tensor):
        h, h0, glob = self.features(x, n_channels)
        logits = self.emit_head(h)
        mask = torch.arange(self.k_max + 1, device=x.device)[None, :] > n_channels[:, None]
        logits = logits.masked_fill(mask[:, None, :], MASK_NEG)
        pooled = h0.mean(dim=1)
        count_logits = self.count_head(torch.cat([pooled, glob], dim=1))
        return logits, count_logits


def kihmm_loss(model: KIHMM, x: torch.Tensor, y: torch.Tensor, n_channels: torch.Tensor,
               w_smooth: float = 1.0, w_emit: float = 0.3, w_count: float = 0.5):
    log_gamma, logZ, logits, count_logits = kihmm_forward(model, x, n_channels)
    smooth = -log_gamma.gather(2, y[:, :, None]).squeeze(-1).mean()
    emit = F.cross_entropy(logits.reshape(-1, K_MAX + 1), y.reshape(-1))
    count = F.cross_entropy(count_logits, (n_channels - 1).long())
    loss = w_smooth * smooth + w_emit * emit + w_count * count
    return loss, {"smooth": float(smooth.detach()), "emit": float(emit.detach()), "count": float(count.detach())}


@torch.no_grad()
def predict_at(model: KIHMM, x: torch.Tensor, n_channels: torch.Tensor):
    model.eval()
    log_gamma, logZ, logits, count_logits = kihmm_forward(model, x, n_channels)
    return log_gamma.argmax(dim=-1), log_gamma.exp(), logZ, count_logits


@torch.no_grad()
def infer_n(model: KIHMM, x: torch.Tensor, mode: str = "count_head", n_candidates=range(1, K_MAX + 1)):
    """Trans-dimensional inference. Returns (N_hat, MAP states, posteriors, logZ per candidate).

    ``count_head`` uses the learned channel-count classifier; ``evidence`` selects N
    by the HMM marginal likelihood (emission scores are normalized per candidate N).
    """
    model.eval()
    Bn = x.shape[0]
    logZ_all = torch.full((Bn, K_MAX + 1), float("-inf"), device=x.device)
    if mode == "count_head":
        _, count_logits = model(x, torch.ones(Bn, dtype=torch.long, device=x.device))
        n_hat = count_logits.argmax(dim=-1) + 1
    else:
        n_hat = torch.ones(Bn, dtype=torch.long, device=x.device)
        best = torch.full((Bn,), -torch.inf, device=x.device)
        for n in n_candidates:
            nc = torch.full((Bn,), n, dtype=torch.long, device=x.device)
            _, logZ, _, _ = kihmm_forward(model, x, nc)
            logZ_all[:, n] = logZ
            take = logZ > best
            n_hat = torch.where(take, nc, n_hat)
            best = torch.where(take, logZ, best)
    log_gamma, logZ, _, _ = kihmm_forward(model, x, n_hat)
    for i in range(Bn):
        logZ_all[i, int(n_hat[i])] = logZ[i]
    return n_hat, log_gamma.argmax(dim=-1), log_gamma.exp(), logZ_all

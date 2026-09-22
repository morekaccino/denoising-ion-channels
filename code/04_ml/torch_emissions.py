"""Learned emission density for a summed patch-clamp trace.

``kinetics.emission_log_densities`` gets log p(y | k open of N) by taking the
generalized-hyperbolic open and closed pdfs from the simulator and convolving
them k and N-k times with an FFT. That is exact but frozen: the shape is fixed
by scipy and there are no gradients.

Here the same quantity is learned. Each channel contributes its level plus a
deviation, and each deviation distribution is a small 1-D Gaussian mixture with
trainable weights, means and widths. Sums of independent Gaussian mixtures
convolve in closed form -- means add, variances add, weights multiply -- so the
k-open density is exact given the learned single-channel shapes, with no FFT and
no grid interpolation:

    log p(y | k, N, s) = logsumexp_c [ log w_c + Normal(y ; m_k + s*mu_c, s*sd_c) ]

where ``c`` ranges over the multisets of k open plus N-k closed components and
``s`` is a per-trace noise scale the network predicts.

The density depends on ``y_t`` and per-trace globals only, never on neighbouring
samples. That is what lets the HMM layer apply the temporal prior exactly once
and makes its log evidence a real likelihood.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import benchmark as B  # noqa: E402
import chain_index as CI  # noqa: E402

LOG2PI = math.log(2.0 * math.pi)
NEG = -1e30


def _multiset_counts(n_comp: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Counts per component for every multiset of size k, plus log multinomial coefficients."""
    picks = CI.sorted_multisets(n_comp, k)
    counts = np.stack([(picks == i).sum(axis=1) for i in range(n_comp)], axis=1).astype(np.float64)
    from scipy.special import gammaln

    log_coef = gammaln(k + 1) - gammaln(counts + 1).sum(axis=1)
    return counts, log_coef


class SumMixtureEmission(nn.Module):
    """log p(y | k open, N-k closed) from learned per-channel mixtures."""

    def __init__(self, n_comp: int = 4, k_max: int = 5,
                 open_level: float = B.OPEN_GH["loc"], close_level: float = B.CLOSE_GH["loc"]):
        super().__init__()
        self.n_comp = n_comp
        self.k_max = k_max
        self.open_level = nn.Parameter(torch.tensor(float(open_level)))
        self.close_level = nn.Parameter(torch.tensor(float(close_level)))
        # a generalized hyperbolic law is a variance-mean mixture of normals, so
        # components are seeded as a geometric ladder of widths around the known
        # per-channel spread with a small mean tilt to break the symmetry
        ladder = torch.logspace(math.log10(0.3), math.log10(3.0), n_comp)
        tilt = torch.linspace(-1.0, 1.0, n_comp)
        self.open_logit = nn.Parameter(torch.zeros(n_comp))
        self.open_mu = nn.Parameter(tilt * 0.02)
        self.open_log_sd = nn.Parameter((ladder * 0.232).log())
        self.close_logit = nn.Parameter(torch.zeros(n_comp))
        self.close_mu = nn.Parameter(tilt * 0.01)
        self.close_log_sd = nn.Parameter((ladder * 0.128).log())
        for N in range(1, k_max + 1):
            self._register_tables(N)

    def _register_tables(self, N: int) -> None:
        """Padded (N+1, C) tables of component counts, one row per open count k."""
        per_k = [(_multiset_counts(self.n_comp, k), _multiset_counts(self.n_comp, N - k))
                 for k in range(N + 1)]
        width = max(len(o[0]) * len(c[0]) for o, c in per_k)
        cnt_o = np.zeros((N + 1, width, self.n_comp))
        cnt_c = np.zeros((N + 1, width, self.n_comp))
        log_coef = np.full((N + 1, width), NEG)
        for k, ((co, lo), (cc, lc)) in enumerate(per_k):
            grid_o = np.repeat(np.arange(len(co)), len(cc))
            grid_c = np.tile(np.arange(len(cc)), len(co))
            n_pair = len(grid_o)
            cnt_o[k, :n_pair] = co[grid_o]
            cnt_c[k, :n_pair] = cc[grid_c]
            log_coef[k, :n_pair] = lo[grid_o] + lc[grid_c]
        self.register_buffer(f"cnt_o_{N}", torch.as_tensor(cnt_o, dtype=torch.float32))
        self.register_buffer(f"cnt_c_{N}", torch.as_tensor(cnt_c, dtype=torch.float32))
        self.register_buffer(f"log_coef_{N}", torch.as_tensor(log_coef, dtype=torch.float32))

    def components(self, N: int):
        """Convolved mixture for each open count k: log weights, deviation mean and
        variance (both (N+1, C)) and the noise-free level (N+1,)."""
        cnt_o = getattr(self, f"cnt_o_{N}")
        cnt_c = getattr(self, f"cnt_c_{N}")
        var_o = (2.0 * self.open_log_sd.clamp(-6.0, 3.0)).exp()
        var_c = (2.0 * self.close_log_sd.clamp(-6.0, 3.0)).exp()
        log_w = (getattr(self, f"log_coef_{N}")
                 + cnt_o @ self.open_logit.log_softmax(-1)
                 + cnt_c @ self.close_logit.log_softmax(-1))
        mu = cnt_o @ self.open_mu + cnt_c @ self.close_mu
        var = (cnt_o @ var_o + cnt_c @ var_c).clamp_min(1e-10)
        k = torch.arange(N + 1, device=mu.device, dtype=mu.dtype)
        return log_w, mu, var, k * self.open_level + (N - k) * self.close_level

    def forward(self, y: torch.Tensor, N: int, log_scale: torch.Tensor,
                chunk: int = 0) -> torch.Tensor:
        """y (B,T), log_scale (B,) -> log p(y_t | k open) of shape (B,T,N+1)."""
        log_w, mu, var, level = self.components(N)
        s = log_scale.exp().clamp(1e-3, 1e3)[:, None, None]
        mean = level[None, :, None] + s * mu[None]
        sd = (s * var[None].sqrt()).clamp_min(1e-6)
        log_norm = log_w[None] - sd.log() - 0.5 * LOG2PI

        step = chunk if chunk > 0 else y.shape[1]
        out = []
        for lo in range(0, y.shape[1], step):
            z = (y[:, lo : lo + step, None, None] - mean[:, None]) / sd[:, None]
            out.append(torch.logsumexp(log_norm[:, None] - 0.5 * z * z, dim=-1))
        return torch.cat(out, dim=1)


def reference_targets(k_max: int, floor: float = 1e-4, n_points: int = 800, device: str = "cpu"):
    """Exact log p(y | k, N) from ``kinetics``, subsampled to the region with mass."""
    import kinetics as K

    out = []
    for N in range(1, k_max + 1):
        grid = K.default_grid(N)
        ref = K.emission_log_densities(N, grid)
        dens = np.exp(ref)
        bulk = dens > floor * dens.max(axis=1, keepdims=True)
        keep = np.flatnonzero(bulk.any(axis=0))
        keep = keep[:: max(1, len(keep) // n_points)]
        w = dens[:, keep]
        out.append({
            "N": N,
            "grid": torch.as_tensor(grid[keep], dtype=torch.float32, device=device),
            "ref": torch.as_tensor(ref[:, keep], dtype=torch.float32, device=device),
            "bulk": torch.as_tensor(bulk[:, keep], device=device),
            "weight": torch.as_tensor(w / w.sum(axis=1, keepdims=True), dtype=torch.float32, device=device),
        })
    return out


def fit_to_simulator(module: SumMixtureEmission, steps: int = 4000, lr: float = 0.03,
                     floor: float = 1e-4, device: str = "cpu",
                     verbose: bool = False) -> SumMixtureEmission:
    """Fit the per-channel mixtures so the convolved densities match the simulator.

    Fitting the single-channel shapes alone is not enough: small tail errors
    compound under convolution, so every (N, k) density is a target. Three terms
    per target -- the forward KL (maximum likelihood under the true density),
    a flat L1 over the region the exact decoder uses, and the worst case on that
    region, which is what the verification reports.
    """
    module = module.to(device)
    targets = reference_targets(module.k_max, floor=floor, device=device)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    zero = torch.zeros(1, device=device)

    for step in range(steps):
        loss, worst = 0.0, 0.0
        for t in targets:
            diff = t["ref"] - module(t["grid"][None], t["N"], zero)[0].T
            on_bulk = diff.abs() * t["bulk"]
            loss = loss + ((t["weight"] * diff).sum(dim=1).mean()
                           + 0.3 * on_bulk.sum(1).div(t["bulk"].sum(1)).mean()
                           + 0.3 * on_bulk.amax(dim=1).mean())
            worst = max(worst, float(on_bulk.amax().detach()))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and step % 500 == 0:
            print(f"  step {step:4d} loss={float(loss.detach()):.5f} worst |dlog p| = {worst:.4f}", flush=True)
    return module


def _verify(n_comp: int = 4, max_n: int = 5, floor: float = 1e-4) -> None:
    """Compare the convolved learned density against the exact FFT reference."""
    import kinetics as K

    torch.manual_seed(0)
    module = fit_to_simulator(SumMixtureEmission(n_comp=n_comp, k_max=max_n), verbose=True)
    module.eval()
    worst = 0.0
    with torch.no_grad():
        for N in range(1, max_n + 1):
            grid = K.default_grid(N)
            ref = K.emission_log_densities(N, grid)
            got = module(torch.as_tensor(grid, dtype=torch.float32)[None], N, torch.zeros(1))[0].T.numpy()
            per_k = []
            for k in range(N + 1):
                mask = np.exp(ref[k]) > floor * np.exp(ref[k]).max()
                per_k.append(float(np.abs(got[k][mask] - ref[k][mask]).max()))
            worst = max(worst, max(per_k))
            print(f"N={N}: max |log p_learned - log p_exact| per k = "
                  f"{[round(v, 4) for v in per_k]}")
    print(f"\nworst over all N and k: {worst:.4f}")
    print("PASS" if worst < 0.05 else f"FAIL (worst {worst:.4f})")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--n-comp", type=int, default=4)
    parser.add_argument("--max-n", type=int, default=5)
    args = parser.parse_args()
    if args.verify:
        _verify(args.n_comp, args.max_n)


if __name__ == "__main__":
    main()

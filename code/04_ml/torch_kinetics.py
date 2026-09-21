"""Differentiable CFTR kinetics: rates -> P0 -> exact count chain -> forward-backward.

Everything ``kinetics.py`` does in numpy/scipy is reproduced here as torch ops so
it can live inside a network and receive gradients:

  rates (12)  --rate_matrix-->  R (7x7)
              --matrix_exp-->   P0 (7x7)
              --chain_index-->  T (SxS), pi_N (S)      [S = C(N+6,6)]
              --forward_backward--> log p(y), posterior over occupancy vectors

Two hot spots are custom autograd Functions with analytic gradients, because the
naive versions either blow up memory (the chain build materialises millions of
intermediate terms) or build a 1000-node graph (the forward-backward loop):

  ``CountTransition``  d T / d logP0 via a scatter-add over the term table;
                       terms are recomputed in the backward pass, so the
                       forward stores nothing but P0.
  ``HMMLogZ``          d logZ / d log b_t = gamma_t, d logZ / d logP = sum xi_t,
                       d logZ / d log pi = gamma_0.

Because ``HMMLogZ`` returns the posterior detached, supervision goes through the
CRF identity instead: log p(states | y) = path_score - logZ, where ``path_score``
is an explicit differentiable gather over the true occupancy trajectory.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import benchmark as B  # noqa: E402
import benchmark_v2 as B2  # noqa: E402
import chain_index as CI  # noqa: E402

N_STATES = 7
N_RATES = len(B2.RATE_INDEX)
OPEN_STATES = np.flatnonzero(B.STATEMAP == 1)
DT = B.DT
CHUNK = 1 << 20


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def rate_matrix(rates: torch.Tensor) -> torch.Tensor:
    """(B,12) off-diagonal rates -> (B,7,7) generator with rows summing to zero."""
    flat = rates.new_zeros(rates.shape[0], N_STATES * N_STATES)
    idx = torch.as_tensor([i * N_STATES + j for i, j in B2.RATE_INDEX], device=rates.device)
    flat.index_copy_(1, idx, rates)
    R = flat.view(-1, N_STATES, N_STATES)
    return R - torch.diag_embed(R.sum(dim=-1))


def p0_from_rates(rates: torch.Tensor, dt: float = DT) -> torch.Tensor:
    return torch.linalg.matrix_exp(dt * rate_matrix(rates))


def stationary(P0: torch.Tensor) -> torch.Tensor:
    """Single-channel stationary distribution: pi P0 = pi, sum pi = 1."""
    eye = torch.eye(N_STATES, device=P0.device, dtype=P0.dtype)
    A = (P0 - eye).transpose(1, 2).clone()
    A[:, -1, :] = 1.0
    rhs = P0.new_zeros(P0.shape[0], N_STATES)
    rhs[:, -1] = 1.0
    return torch.linalg.solve(A, rhs).clamp_min(1e-12)


class ChainTables:
    """Device-resident term table and state metadata for one channel count N."""

    def __init__(self, N: int, device: str | torch.device = "cpu"):
        t = CI.load_chain_index(N)
        states = t["states"].astype(np.int64)
        self.N = N
        self.size = len(states)
        self.device = torch.device(device)
        self.picks = torch.as_tensor(t["picks"].astype(np.int64), device=device)
        self.log_coef = torch.log(torch.as_tensor(t["coef"], dtype=torch.float32, device=device))
        self.flat_rc = torch.as_tensor(
            (t["rows"].astype(np.int64) * self.size + t["cols"]), device=device
        )
        self.states = torch.as_tensor(states, dtype=torch.float32, device=device)
        self.log_multinom = torch.as_tensor(t["log_multinom"], dtype=torch.float32, device=device)
        self.open_count = torch.as_tensor(states[:, OPEN_STATES].sum(axis=1), device=device)
        base = N + 1
        powers = base ** np.arange(N_STATES)
        lut = np.full(base**N_STATES, -1, dtype=np.int64)
        lut[states @ powers] = np.arange(len(states))
        self.lut = torch.as_tensor(lut, device=device)
        self.lut_powers = torch.as_tensor(powers, device=device)

    def state_index(self, counts: torch.Tensor) -> torch.Tensor:
        """(B,7,T) occupancy counts -> (B,T) row index into the chain."""
        return self.lut[(counts.long() * self.lut_powers[None, :, None]).sum(dim=1)]


_TABLE_CACHE: dict[tuple[int, str], ChainTables] = {}


def get_tables(N: int, device: str | torch.device) -> ChainTables:
    key = (int(N), str(device))
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = ChainTables(int(N), device)
    return _TABLE_CACHE[key]


class CountTransition(torch.autograd.Function):
    """logP0 (B,49) -> exact occupancy-count transition matrix (B,S,S)."""

    @staticmethod
    def forward(ctx, logP0_flat: torch.Tensor, tables: ChainTables) -> torch.Tensor:
        Bn, S, M = logP0_flat.shape[0], tables.size, tables.picks.shape[0]
        T = logP0_flat.new_zeros(Bn, S * S)
        with torch.no_grad():
            for lo in range(0, M, CHUNK):
                hi = min(lo + CHUNK, M)
                acc = tables.log_coef[lo:hi].expand(Bn, hi - lo).clone()
                for u in range(tables.N):
                    acc = acc + logP0_flat[:, tables.picks[lo:hi, u]]
                T.index_add_(1, tables.flat_rc[lo:hi], acc.exp_())
        ctx.save_for_backward(logP0_flat)
        ctx.tables = tables
        return T.view(Bn, S, S)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (logP0_flat,) = ctx.saved_tensors
        tables: ChainTables = ctx.tables
        Bn, M = logP0_flat.shape[0], tables.picks.shape[0]
        g_flat = grad_out.reshape(Bn, -1)
        grad = torch.zeros_like(logP0_flat)
        for lo in range(0, M, CHUNK):
            hi = min(lo + CHUNK, M)
            acc = tables.log_coef[lo:hi].expand(Bn, hi - lo).clone()
            for u in range(tables.N):
                acc = acc + logP0_flat[:, tables.picks[lo:hi, u]]
            w = acc.exp_() * g_flat[:, tables.flat_rc[lo:hi]]
            for u in range(tables.N):
                grad.index_add_(1, tables.picks[lo:hi, u], w)
        return grad, None


def count_transition(P0: torch.Tensor, tables: ChainTables) -> torch.Tensor:
    return CountTransition.apply(P0.clamp_min(1e-30).log().reshape(P0.shape[0], -1), tables)


def count_log_initial(pi_single: torch.Tensor, tables: ChainTables) -> torch.Tensor:
    """(B,7) single-channel stationary -> (B,S) log multinomial occupancy prior."""
    return tables.log_multinom[None, :] + pi_single.log() @ tables.states.T


class HMMLogZ(torch.autograd.Function):
    """Scaled forward-backward. Returns log evidence; posterior comes out detached."""

    @staticmethod
    def forward(ctx, logb: torch.Tensor, logP: torch.Tensor, logpi: torch.Tensor):
        with torch.no_grad():
            Bn, T, S = logb.shape
            shift = logb.amax(dim=-1)
            b = (logb - shift[..., None]).exp()
            P, pi = logP.exp(), logpi.exp()

            alpha = torch.empty_like(b)
            scale = b.new_empty(Bn, T)
            a = pi * b[:, 0]
            scale[:, 0] = a.sum(-1).clamp_min(1e-300)
            alpha[:, 0] = a / scale[:, 0, None]
            for t in range(1, T):
                a = torch.bmm(alpha[:, t - 1].unsqueeze(1), P).squeeze(1) * b[:, t]
                scale[:, t] = a.sum(-1).clamp_min(1e-300)
                alpha[:, t] = a / scale[:, t, None]

            beta = torch.empty_like(b)
            beta[:, T - 1] = 1.0
            for t in range(T - 2, -1, -1):
                nxt = (b[:, t + 1] * beta[:, t + 1]) / scale[:, t + 1, None]
                beta[:, t] = torch.bmm(P, nxt.unsqueeze(2)).squeeze(2)

            gamma = alpha * beta
            gamma = gamma / gamma.sum(-1, keepdim=True).clamp_min(1e-30)
            logZ = scale.log().sum(-1) + shift.sum(-1)

        ctx.save_for_backward(alpha, beta, b, scale, P, gamma)
        ctx.mark_non_differentiable(gamma)
        return logZ, gamma

    @staticmethod
    def backward(ctx, grad_logZ: torch.Tensor, _grad_gamma):
        alpha, beta, b, scale, P, gamma = ctx.saved_tensors
        g = grad_logZ[:, None, None]
        grad_logb = gamma * g
        grad_logpi = gamma[:, 0] * grad_logZ[:, None]
        D = b[:, 1:] * beta[:, 1:] / scale[:, 1:, None]
        grad_logP = torch.einsum("bti,btj->bij", alpha[:, :-1], D) * P * g
        return grad_logb, grad_logP, grad_logpi


def hmm_posterior(logb: torch.Tensor, logP: torch.Tensor, logpi: torch.Tensor):
    """log evidence (B,) and posterior over occupancy vectors (B,T,S), detached."""
    return HMMLogZ.apply(logb, logP, logpi)


def path_score(
    logb: torch.Tensor, logP: torch.Tensor, logpi: torch.Tensor, path: torch.Tensor
) -> torch.Tensor:
    """log p(y, states) along a known occupancy trajectory. path: (B,T) state indices."""
    emit = logb.gather(2, path[:, :, None]).squeeze(-1).sum(-1)
    start = logpi.gather(1, path[:, :1]).squeeze(-1)
    step = logP.gather(1, path[:, :-1, None].expand(-1, -1, logP.shape[-1]))
    trans = step.gather(2, path[:, 1:, None]).squeeze(-1).sum(-1)
    return emit + start + trans


def expand_open_emissions(log_emit_k: torch.Tensor, tables: ChainTables) -> torch.Tensor:
    """(B,T,N+1) emissions indexed by open count -> (B,T,S) indexed by occupancy state."""
    return log_emit_k.index_select(2, tables.open_count)


def posterior_counts(gamma: torch.Tensor, tables: ChainTables) -> torch.Tensor:
    """(B,T,S) posterior -> (B,7,T) expected channels per kinetic state."""
    return (gamma @ tables.states).transpose(1, 2)


def _verify(max_n: int = 5, device: str = "cpu") -> None:
    import kinetics as K
    from scipy.linalg import expm

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    worst = {"P0": 0.0, "pi": 0.0, "T": 0.0, "piN": 0.0, "gamma": 0.0, "logZ": 0.0, "grad": 0.0}

    for N in range(1, max_n + 1):
        tables = get_tables(N, device)
        rates_np = B2.BASE_OFF * np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=(2, N_RATES)))
        rates = torch.as_tensor(rates_np, dtype=torch.float64, device=device)

        P0 = p0_from_rates(rates)
        pi1 = stationary(P0)
        for i in range(2):
            ref_P0 = expm(DT * B2.rate_table(rates_np[i]))
            worst["P0"] = max(worst["P0"], float(np.abs(P0[i].cpu().numpy() - ref_P0).max()))
            worst["pi"] = max(worst["pi"], float(np.abs(pi1[i].cpu().numpy() - B.stationary(ref_P0)).max()))

        Tm = count_transition(P0.float(), tables)
        logpiN = count_log_initial(pi1.float(), tables)
        for i in range(2):
            ref_P0 = expm(DT * B2.rate_table(rates_np[i]))
            worst["T"] = max(worst["T"], float(np.abs(Tm[i].cpu().numpy() - K.count_transition_matrix(N, ref_P0)).max()))
            worst["piN"] = max(worst["piN"], float(np.abs(logpiN[i].exp().cpu().numpy() - K.count_stationary(N, ref_P0)).max()))

        # forward-backward against the numpy reference, on random emissions
        S, T_len = tables.size, 60
        logb = torch.as_tensor(rng.normal(size=(2, T_len, S)), dtype=torch.float32, device=device)
        logb.requires_grad_(True)
        logP, logpi = Tm.clamp_min(1e-30).log(), logpiN
        logZ, gamma = hmm_posterior(logb, logP, logpi)
        for i in range(2):
            ref_P0 = expm(DT * B2.rate_table(rates_np[i]))
            _, _, _, g_ref, z_ref = K.forward_backward(
                logb[i].detach().cpu().numpy().astype(float),
                K.count_transition_matrix(N, ref_P0),
                K.count_stationary(N, ref_P0),
            )
            worst["gamma"] = max(worst["gamma"], float(np.abs(gamma[i].cpu().numpy() - g_ref).max()))
            worst["logZ"] = max(worst["logZ"], abs(float(logZ[i].detach()) - z_ref) / max(abs(z_ref), 1.0))

        # the analytic gradient of logZ wrt log emissions must equal the posterior
        logZ.sum().backward()
        worst["grad"] = max(worst["grad"], float((logb.grad - gamma).abs().max()))

        print(f"N={N}: P0={worst['P0']:.1e} pi={worst['pi']:.1e} T={worst['T']:.1e} "
              f"piN={worst['piN']:.1e} gamma={worst['gamma']:.1e} logZ={worst['logZ']:.1e}")

    # gradcheck the chain build and the HMM against finite differences (N=2, float64)
    tables64 = ChainTables(2, device)
    tables64.log_coef = tables64.log_coef.double()
    S = tables64.size
    lp = torch.as_tensor(np.log(expm(DT * B2.rate_table(B2.BASE_OFF))).reshape(1, -1),
                         dtype=torch.float64, device=device).requires_grad_(True)
    ok_chain = torch.autograd.gradcheck(
        lambda z: CountTransition.apply(z, tables64), (lp,), eps=1e-6, atol=1e-6
    )
    logb = torch.as_tensor(rng.normal(size=(1, 12, S)), dtype=torch.float64,
                           device=device).requires_grad_(True)
    logP = CountTransition.apply(lp.detach(), tables64).clamp_min(1e-30).log().requires_grad_(True)
    logpi = torch.as_tensor(rng.normal(size=(1, S)), dtype=torch.float64,
                            device=device).log_softmax(-1).requires_grad_(True)
    ok_hmm = torch.autograd.gradcheck(
        lambda a, b_, c: HMMLogZ.apply(a, b_, c)[0], (logb, logP, logpi), eps=1e-6, atol=1e-6
    )
    print(f"\ngradcheck: chain={ok_chain} hmm={ok_hmm}")
    tol = {"P0": 1e-6, "pi": 1e-6, "T": 1e-6, "piN": 1e-6, "gamma": 1e-4, "logZ": 1e-5, "grad": 1e-5}
    bad = {k: v for k, v in worst.items() if v > tol[k]}
    print("PASS" if not bad and ok_chain and ok_hmm else f"FAIL {bad}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--max-n", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.verify:
        _verify(args.max_n, args.device)


if __name__ == "__main__":
    main()

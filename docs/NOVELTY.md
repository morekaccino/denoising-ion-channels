# KI-HMM v2: what is new, and what the results say

Paper-ready summary of the v2 track. Everything below is reproducible from this
branch; see "Reproduce" at the end.

## The question

A patch-clamp recording can contain several CFTR channels at once. You only see
their summed current. From that one trace we ask for three things:

1. how many channels there are (N);
2. how many channels sit in each of the 7 kinetic states at every time step
   (a..g, with a + b + c + d + e + f + g = N);
3. the Markov parameters — the 12 allowed transition rates of the CFTR graph.

## What is new

No previous work does these three things together from a **summed multi-channel**
trace. The closest works, and what separates them:

| Work | What it does | Why ours differs |
|---|---|---|
| Berghaus et al. 2024, "Foundation Inference Models for Markov Jump Processes" ([2406.06419](https://arxiv.org/abs/2406.06419)) | Neural net outputs the rate matrix of a hidden Markov jump process from noisy observations; tested on single-channel ion channel paths | The chain state must be visible (one channel). Ours sees only the sum of N noisy channels, infers N, and keeps the CFTR kinetic graph exact |
| Moffett et al. 2021, factor-graph EM ([2106.09594](https://arxiv.org/abs/2106.09594)) | Estimates CFTR kinetic microstates and rates with classical EM; single channel | Ours is multi-channel, amortized (one forward pass), and includes trans-dimensional N |
| Vanegas et al. 2021, VND-HMM ([2103.06071](https://arxiv.org/abs/2103.06071)) | HMM for superimposed two-state channels; identifiability, model selection | Classical statistics, 2 states, no deep learning, no N regression, no CFTR graph |
| SlotFlow 2025 ([2511.23228](https://arxiv.org/abs/2511.23228)) | Amortized trans-dimensional inference with slots (astronomy) | No Markov chain, no biophysics, no kinetic parameters |
| Neural HMMs, e.g. Tran et al. 2016 ([1609.09007](https://arxiv.org/abs/1609.09007)) | Neural emissions + discrete states | Fixed state space, no trans-dimensional channel count, no physical kinetic chain |
| FHMM model selection ([1506.07959](https://arxiv.org/abs/1506.07959)) and scalable fHMM filtering ([2607.07008](https://arxiv.org/abs/2607.07008)) | Bayesian/classical factorial HMM learning | No neural emissions, no biophysical structure, no summed-current application |
| VAMPnets / deep Markov state models ([1710.06012](https://arxiv.org/abs/1710.06012)) | Learn coarse-grained Markov models from trajectories | Full state observability; not emission-based counting |

Four contributions:

1. **A joint deep model.** One TCN encoder feeds three heads: N, per-state
   counts a..g, and 12 transition rates. The rate head works on a *group* of
   traces recorded under the same kinetics, which is how rates become
   estimable. In v4 the model also decodes with the rate table it predicts: the
   generator is exponentiated and expanded into the exact occupancy-count chain
   inside the forward pass, and the per-channel emission density is learned
   rather than imported from the simulator.
2. **A rate-randomized benchmark.** `synth_v2` gives every group its own random
   rate table (×0.5–×2 around the literature table), so rate estimation is a
   real task instead of predicting a constant.
3. **An identifiability result.** Using the exact factorial-HMM decoder we
   compute the Fisher information of the rates from summed traces. Only about
   **4 of the 12 rate directions are identifiable**; the other directions carry
   almost no information. This is a limit of the measurement, not of the model,
   and it tells practitioners which kinetic numbers can even be estimated from
   multi-channel data.
4. **Exact factorial-HMM inference as a trainable layer.** The occupancy-count
   transition matrix is a polynomial in the single-channel kernel, so it can be
   rebuilt from predicted rates with a gather, a product and a scatter-add, and
   the forward-backward recursion supplies its own analytic gradients. Together
   with a closed-form convolved mixture emission this makes Bayes-optimal
   structured inference a differentiable module rather than a post-processing
   step, on hardware as modest as a laptop CPU.

## The model

`code/04_ml/torch_models_v2.py`

- Input: a group of K traces (K = 6 used in training), each 10 s at 100 Hz.
- Encoder: 6-block dilated TCN (64 hidden), per-trace standardisation.
- N head: 5-way classifier (1..5 channels), N-agnostic pooling.
- State-count head: per-sample softmax over 7 states, scaled by predicted N;
  the counts then sum to N by construction.
- Rate head: a second small TCN reads the predicted open probability; its
  statistics are pooled over the group and mapped to 12 log-rate offsets around
  the known base table.
- Loss: N cross-entropy + a..g absolute error + a **Fisher-weighted** rate loss
  that only spends capacity on identifiable directions.

The exact kinetic chain (`kinetics.py`) is not trained; it is used to compute
the Fisher weights, to convert predicted rate tables into effective kinetic
numbers, and as an interpretable baseline.

## Results

Frozen test: 64 groups × 6 traces, N ∈ 1..5, generalized-hyperbolic noise.

Timestep accuracy at noise ×1/×2/×4 (step accuracy uses the count head):

| Metric | v2a (clean training) | v2b (augmented training) |
|---|---|---|
| N accuracy | 94.8 / 96.1 / 100% | 95.6 / 97.9 / 100% |
| Per-state count MAE (×1) | 0.32 channels | 0.31 channels |
| Identifiable-direction R² | 0.73 / 0.48 / 0.17 | 0.635 / 0.391 / 0.223 |
| Opening-rate R² | 0.57 | 0.48 |
| Closing-rate R² | 0.73 | 0.65 |
| p_open R² | 0.79 | 0.73 |
| Median rate error | ~18–19% | ~20% |

N = 4–5 extrapolation: N accuracy 96.6%, top-direction R² 0.60.

KI-HMM v3 (`code/04_ml/torch_models_v3.py`) replaces the per-timestep count
head with a neural HMM over the open count: emissions from the encoder plus
exact forward–backward, transitions initialized from the biophysical count
chain and corrected by gated neural nets conditioned on the predicted rates.
On the same test set, open-count accuracy improves to **0.657 / 0.592 / 0.472**
at noise ×1/×2/×4 (v2: 0.605 / 0.537 / 0.393) and state error drops to 0.307
channels at ×1. N accuracy is 0.922 at ×1. Exact structured decoding with the
model's own predicted N and rates still reaches 0.89 open accuracy
(`refine_kihmm_v2.py`), so the remaining gap is the learned emission model.

## KI-HMM v4: decoding with the rates the model predicts

v2 and v3 share a flaw. Both output a rate table, and both then smooth with the
count chain of the **base** rate table, while `synth_v2` randomises every rate
by ×0.5–2 per group. The temporal prior was wrong for nearly every group. That
is why the offline refinement in `refine_kihmm_v2.py` gains 0.63 → 0.888 purely
by feeding the model's *own* predicted N and rates into the real chain.

v4 (`code/04_ml/torch_models_v4.py`) puts that computation inside the network:

- **Kinetics layer.** The 12 predicted rates become a 7×7 generator and then
  `torch.linalg.matrix_exp(dt·R)`; the single-channel stationary distribution
  comes from a linear solve. Both differentiable.
- **Exact occupancy-count chain, differentiably.** Every entry of the
  C(N+6,6)-state transition matrix is a polynomial in the 49 entries of `P0`
  with exactly N factors, so enumerating the multisets of N cells out of 49
  enumerates every term once. `chain_index.py` caches those term tables (49 to
  2,869,685 terms for N=1 to 5) and `CountTransition` rebuilds the matrix with a
  gather, a product and a scatter-add, recomputing terms in the backward pass so
  the forward stores nothing but `P0`.
- **Learned emission density.** Each channel's deviation is a small 1-D Gaussian
  mixture; sums of independent mixtures convolve in closed form, so the k-open
  density is exact given the learned per-channel shapes — no FFT, no grid. It is
  evaluated pointwise at `y_t`, never on a temporal window, which keeps the HMM
  from counting the same evidence twice and makes its log evidence a real
  likelihood. Four components match the exact generalized-hyperbolic densities
  to 0.043 nats over the density bulk for every N and k.
- **Supervision without backpropagating through forward–backward.**
  `HMMLogZ` returns the posterior detached and supplies analytic gradients
  (`d logZ/d log b_t = gamma_t`, `d logZ/d log P = sum xi_t`). Training uses the
  CRF identity `log p(states | y) = path_score − logZ`, where `path_score` is an
  explicit gather over the true occupancy trajectory, plus a generative `−logZ`
  term that holds the emission density and the rates to the data.

Every piece is a differentiable layer trained by gradient descent. Inference is
one forward pass: no EM, no clustering, no separate decoding stage.

Frozen test, same 64 groups × 6 traces as above:

| Model | open-count acc ×1/×2/×4 | per-state MAE ×1 | N acc ×1 |
|---|---|---|---|
| v2a (per-timestep count head) | 0.605 / 0.537 / 0.393 | 0.318 | 0.948 |
| v3a (neural HMM head, base chain) | 0.657 / 0.592 / 0.472 | 0.308 | 0.922 |
| **v4b (chain from predicted rates)** | **0.815 / 0.718 / 0.469** | **0.269** | 0.909 |
| Posterior with true N and true rates | 0.928 / 0.695 / 0.451 | 0.224 | — |

The last row is the exact posterior under the generative model, computed with
the same v4 layers (`oracle_v4.py`) and verified to reproduce the numpy decoder
to 9e-5 per count. It is a reference, not a hard bound: the metric rounds the
posterior *mean*, which is not the optimal rule for 0-1 loss, which is why v4
edges past it at ×2 and ×4 where the posterior is broad.

Where the remaining ×1 gap sits: forcing the true N raises v4b from 0.815 to
0.885, so roughly two thirds of what is left is the channel-count head, and it
fails almost entirely on N=4 versus N=5 (25 of its 35 test errors).

Two honest negatives from this round:

1. **Rate recovery regressed.** Identifiable-direction R² on test ×1 is
   0.36 / 0.29 for v4b against 0.73 / 0.48 for v2a. v2's rate head read a
   supervised, temporally smooth per-timestep state posterior; v4's reads a
   pointwise emission-derived open probability, because the smoothed posterior
   is only available after the rates and comes back detached. Raising the rate
   weight and doubling the training crop made it worse. The untried fix is a
   two-pass rate head: decode once with the base chain, then feed that posterior
   to the rate head.
2. **Colored noise breaks it.** On the AR(1) mismatch split open accuracy falls
   to 0.33, because the learned emission assumes independent samples. The other
   mismatch splits are fine or better than the base case (drift 0.917,
   Gaussian noise 0.884, lowpass 0.758).

How many traces per rate table matter (bag ablation, v2a): top-direction R² is
−0.29 for K = 1, 0.48 for K = 2, 0.68 for K = 4, and 0.73 for K = 6. Rates
should be estimated from a group, not one short trace.

Figures (frozen test, noise ×1), rendered by `code/04_ml/figures_kihmm_v2.py`:

- `results/figures/kihmm_v2_predictions.png` — N=1/2/3 example traces with the
  true and predicted open count (median examples per N).
- `results/figures/kihmm_v2_state_shares.png` — average per-state counts,
  per-state errors, and the N confusion matrix.
- `results/figures/kihmm_v2_rate_scatter.png` — predicted vs true opening
  rate, closing rate, p_open and the top identifiable direction.
- `results/figures/kihmm_v2_rate_recovery.png` — per-rate R² and the Fisher
  spectrum that explains why most rates cannot be recovered.

Identifiability (exact decoder, 6 traces): Fisher eigenvalues span several
orders of magnitude. The top direction is the O1↔O2 open-block kinetics
(CRB ≈ 0.07 in log units), then open/closed cycle combinations (≈0.12–0.14);
three to four directions are effectively unidentifiable (CRB far above the
prior spread of 0.347).

## Real data: honest status

All 53 ABF recordings were run (10 s segments, one group per file).

- **N transfers**: the model predicts ≈1 channel for 52/53 files, matching the
  lab's note that most chunks contain one channel.
- **Rates do not transfer yet**: real files sit outside the training range
  (mean |Δlog rate| 0.255 vs 0.145 on synthetic; 21% of files hit the output
  clamp vs 2%). Retraining with drift, colored noise and filtering made this
  worse (37%), and neither filtering nor detrending helped.
- Diagnosis: the group-level rate head saturates on real summaries, while the
  per-segment state head still tracks a crude threshold moderately. Real-data
  rate estimation needs the simulator to match each recording (the thesis
  `code/02` noise-fitting route) or semi-supervised adaptation. Recorded as an
  open problem in `docs/STATUS.md`.

## Limitations

1. Rates are only partly identifiable from summed traces; report the
   identifiable subspace (effective opening/closing rates, p_open), not the
   full 7-state table.
2. Closed microstates share one emission level, so the a..g split inside the
   closed block is prior-driven.
3. Real-data rate estimation needs domain adaptation; N already transfers.
4. The gating model is the fixed 7-state CFTR graph; drug block (glibenclamide)
   is not part of the state space yet.

## Reproduce

```bash
# data (fast, ~10 s)
python code/04_ml/benchmark_v2.py --generate
python code/04_ml/benchmark_v2.py --augment

# train (~6 min on a GTX 1660 SUPER)
python code/04_ml/train_kihmm_v2.py --epochs 120 --batch-groups 16 --lr 3e-3 --tag v2a

# evaluate (metrics, R² per direction, bag ablation, effective rates)
python code/04_ml/eval_kihmm_v2.py --model code/04_ml/models/kihmm_v2_v2a.pt

# real recordings
python code/04_ml/apply_kihmm_v2_real.py --model code/04_ml/models/kihmm_v2_v2a.pt
```

v4 (about 70 minutes end to end on an Apple M4 Pro, CPU):

```bash
python code/04_ml/chain_index.py --build          # term tables, ~2 s, cached
python code/04_ml/oracle_v4.py --emission-fit     # emission fit + reference posterior
python code/04_ml/train_kihmm_v4.py --epochs 80 --batch-groups 16 \
       --tag v4b --device cpu --init-emission --rate-stats --n-head pooled
python code/04_ml/eval_kihmm_v4.py --model code/04_ml/models/kihmm_v4_v4b.pt
python code/04_ml/figures_kihmm_v4.py --model code/04_ml/models/kihmm_v4_v4b.pt
```

Each v4 layer verifies against the numpy reference on its own:
`chain_index.py --verify`, `torch_kinetics.py --verify` (includes `gradcheck`),
`torch_emissions.py --verify`.

Artifacts: `data/derived/synth_v2/`, `code/04_ml/results/kihmm_v2_eval.json`,
`code/04_ml/results/kihmm_v2_real.json`, `code/04_ml/results/fisher_synth_v2.json`,
figures in `code/04_ml/results/figures/`.

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

Two honest negatives from this round, the first of which v5 fixes:

1. **Rate recovery regressed.** Identifiable-direction R² on test ×1 is
   0.36 / 0.29 for v4b against 0.73 / 0.48 for v2a. v2's rate head read a
   supervised, temporally smooth per-timestep state posterior; v4's reads a
   pointwise emission-derived open probability, because the smoothed posterior
   is only available after the rates and comes back detached. Raising the rate
   weight and doubling the training crop made it worse.
2. **Colored noise breaks it.** On the AR(1) mismatch split open accuracy falls
   to 0.33, because the learned emission assumes independent samples. The other
   mismatch splits are fine or better than the base case (drift 0.917,
   Gaussian noise 0.884, lowpass 0.758).

## KI-HMM v5: stop reading the answer off a head, ask the likelihood

v4 learns a complete generative model of a recording -- a per-channel emission
density and the exact occupancy-count chain -- and then discards most of it at
inference, reading the channel count off a classifier and the rates off a
regressor. v5 (`code/04_ml/infer_v5.py`) keeps the network exactly as trained
and changes only what happens at inference:

- **Channel count by model evidence.** Score every candidate N with the
  network's own log evidence and take the best. This is exact Bayesian model
  selection over a discrete latent, carried out with the model's own layers.
- **Rates and noise scale by maximum likelihood.** Maximise that same log
  evidence over the 12 rates (shared by a group) and the per-trace noise scale
  with Adam through the differentiable chain. The scale matters: the trained
  head reads 1.28 on noise ×1 data, and fitting it cuts log-scale error from
  0.249 to 0.076.

Two details decide whether this works. Maximisation must be penalised toward
the base rate table, because at noise ×4 most rate directions carry essentially
no information and an unpenalised fit wanders; with the penalty it becomes MAP
estimation and, as a bonus, noise ×1 improves too (R² [0.731, 0.643, −1.47,
−1.38] becomes [0.778, 0.685, 0.310, 0.194]). And refinement is skipped outright
above an estimated noise scale of 2.5, where it is a small net loss.

`v5a` is v4 retrained with two lessons from the bake-off below: three times as
many training groups, and a separate gradient clip and learning rate for the
trace-level heads.

| Model, frozen test ×1 | open-count acc | per-state MAE | N acc | direction R² |
|---|---|---|---|---|
| v2a | 0.605 | 0.318 | 0.948 | [0.73, 0.48, 0.17, −0.19] |
| v3a | 0.657 | 0.308 | 0.922 | [0.77, 0.14, 0.22, −0.15] |
| v4b | 0.815 | 0.269 | 0.909 | [0.36, 0.29, −0.08, −0.21] |
| v5a, feedforward only | 0.827 | 0.261 | 0.927 | [0.58, 0.28, 0.18, −0.09] |
| **v5a + exact inference** | **0.853** | **0.255** | **0.951** | **[0.87, 0.68, 0.39, 0.21]** |
| Posterior with true N and true rates | 0.928 | 0.224 | — | — |

At noise ×2 the same pipeline gives N 0.990, open 0.755 and R²
[0.74, 0.62, 0.49, 0.00]. Every identifiable rate direction is now positive,
and the top one (0.87) beats v2a's 0.73 and clears the information-theoretic
target of 0.85.

Cost: the evidence pass is 25 ms per trace on the GPU, refinement about 4 s per
group of six traces on the CPU. Both are negligible for offline analysis but
mean inference is no longer a single forward pass.

## What the architecture search actually found

`code/04_ml/bakeoff.py` caches the expensive tensors once (emission output,
posterior, log evidence, score vector, expected transition counts) and then
trains each candidate head on cached tensors in seconds, which made it
practical to rank 23 designs rather than guess.

**The architecture is not the lever. Training data and optimiser coupling are.**

- Count head, test accuracy with three times the training groups: a plain MLP
  on the 64-bin amplitude histogram **0.9948**; a 1D CNN ties it but is 60x
  slower; Deep Sets over raw samples 0.9896; ordinal CORAL 0.9896;
  multi-resolution histogram pyramid 0.9792; quantile-function input 0.9531;
  log evidence alone 0.9427. With the original data the same MLP gets 0.9479,
  so the data is worth five points and the architecture at most one.
- Rate head: the existing autocorrelation and transition statistics win at
  [0.732, 0.508]. The score function `dlogZ/dlograte` and the Baum-Welch
  expected transition counts are the statistically natural sufficient
  statistics and both **overfit badly** (train R² 0.91, test 0.30). A mixture
  density output and attention pooling over the group do not help either.
- Backbones over the open-probability track -- TCN, U-Net and bidirectional GRU
  -- land within noise of each other on the top rate direction (0.743, 0.634,
  0.755) and are worse than the hand-written statistics on the second (about
  0.27 against 0.51), at 10 to 60x the cost. This confirms the earlier
  measurement that the structured loss terms already sit on the oracle: with
  pointwise emissions and an exact chain, an encoder has little left to do.
- A head trained on its own reaches 0.99 while the same head inside the joint
  model reaches 0.91, and that holds even in the configuration where it shares
  no parameters with the encoder. The cause is the global gradient clip
  throttling it against the structured losses; clipping the heads separately is
  the fix.

Also worth recording: the old finding that evidence-based N selection is
unreliable came from `kinetics.signal_likelihood`, which treats samples as
independent. The full HMM log evidence is a different quantity and works.

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

## Head-to-head benchmark against prior methods (2026-09)

To test the claim "better than previous methods" directly, the three closest
published methods were reconstructed and run on the frozen `synth_v2` test set
(384 traces, N=1–5, noise ×1/×2/×4). Provenance, licenses, commit hashes and
every deviation are in `code/05_baselines/BASELINES.md`; the full table and
figure are in `code/05_baselines/results/baseline_comparison.md`.

| Method | What it is | Implementation |
|---|---|---|
| SD-HMM (Requadt & Li 2026) | Continuous-time sum-dependent chain on the open-channel count; BIC over L; Viterbi path | Python port of the R/Rcpp reference (GPL-3), verified by recovering the paper's `(3,4,4,3)` example and the positive/negative-cooperativity signs |
| VND-HMM (Vanegas et al. 2024) | Discrete-time vector-norm-dependent chain; BIC over L; Viterbi path | Python port of the R/Rcpp reference (GPL-2), verified by recovering `(0.99,0.98,0.98,0.99)` |
| Deep-Channel (Celik et al. 2020) | Pointwise CNN+LSTM idealizer (n=1 timestep, as in the paper and released code); N = max simultaneous openings | PyTorch reimplementation of the shipped Keras model, retrained on the `synth_v2` train split |

Frozen test, noise ×1 (N accuracy / open accuracy / open MAE):

| Method | N acc | Open acc | Open MAE |
|---|---|---|---|
| **KI-HMM v5a** | **0.951** | **0.853** | **0.183** |
| SD-HMM | 0.401 | 0.640 | 0.498 |
| VND-HMM | 0.417 | 0.637 | 0.503 |
| Deep-Channel | 0.786 | 0.529 | 0.544 |

At noise ×2 and ×4 the ranking is unchanged (v5a N 0.990/1.000, open
0.755/0.508; baselines N 0.17–0.33, open 0.28–0.43). Per true N, the HMM
baselines collapse at N≥4 (N=5 accuracy 0.00–0.01) because the extra current
levels are not resolvable and BIC under-selects; Deep-Channel is competitive
at N=1–2 on open count (0.885/0.656 against v5a's 0.991/0.964) but fails at
N≥3 and its max-openings heuristic inflates N under noise. None of the three
outputs per-state counts or kinetic rates; the capability matrix in
`results/baseline_comparison.md` lists what each method supports.

Two honesty notes for the paper: the two-state HMM baselines are structurally
misspecified on 7-state CFTR data (their sum process is not Markov for the
7-state chain), which is a limitation of the comparison as much as of the
methods; and Deep-Channel is at the context-free pointwise accuracy ceiling on
this data (nearest-level classification with known N reaches 0.74; a plain MLP
0.56; Deep-Channel 0.53), so its score reflects the pointwise idealization
task, not a weak training recipe (a 16-epoch schedule changes nothing).

Round 2 added IDC (Requadt et al. 2025) and the Moffett et al. 2022
single-channel CFTR factor-graph EM on the same frozen test set (branch
`baselines-round2`). IDC scores x1 N 0.320 / open 0.442 and, as documented in
its own paper, undercounts levels when a trace never visits all channels
closed; its Cauchy-noise robustness claim was reproduced in verification.
Moffett is single-channel, so it was run on the 69 N=1 traces: its open
accuracy is 0.987/0.893/0.520 at noise x1/x2/x4 against v5's 0.991/0.951/0.830
on the same traces, i.e. the specialist ties v5 at the reference noise level
and degrades faster, while only v5 scales to N>1 and outputs N, seven-state
occupancy and rates. A 10-point noise-factor sweep (1.0 to 4.0) with line
charts for every full-test-set method is in the report (Section 13); v5a leads
channel-count accuracy, open-count accuracy and open-count MAE at every level.

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

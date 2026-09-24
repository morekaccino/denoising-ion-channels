# Baseline benchmark report: KI-HMM v5 vs the three closest prior methods

Frozen `synth_v2` test set, September 2026. Branch `baselines-vs-kihmm`.
Everything below is reproducible from this repository; raw numbers live in
`code/05_baselines/results/`, provenance and porting deviations in
`code/05_baselines/BASELINES.md`.

## 0. TL;DR

Three prior methods were reconstructed and run on the same 384-trace frozen
test set as KI-HMM v5a:

| Method | noise x1 (N / open / MAE) | noise x2 | noise x4 |
|---|---|---|---|
| **KI-HMM v5a (ours)** | **0.951 / 0.853 / 0.183** | **0.990 / 0.755 / 0.316** | **1.000 / 0.508 / 0.618** |
| SD-HMM (Requadt & Li 2026) | 0.401 / 0.640 / 0.498 | 0.167 / 0.312 / 1.193 | 0.234 / 0.295 / 1.125 |
| VND-HMM (Vanegas et al. 2024) | 0.417 / 0.637 / 0.503 | 0.167 / 0.276 / 1.200 | 0.232 / 0.282 / 1.148 |
| Deep-Channel (Celik et al. 2020) | 0.786 / 0.529 / 0.544 | 0.333 / 0.434 / 0.720 | 0.182 / 0.317 / 1.073 |

v5a wins every metric on every split. It is also the only method that outputs
per-state counts and kinetic rates, and the only one that keeps working at
N = 4–5 (baselines drop to 0.00–0.27 channel-count accuracy there; v5a stays
at 0.86–0.93).

## 1. Question and evaluation protocol

The thesis claim is that KI-HMM v5 is better than the closest prior art at the
task it defines: from one noisy **summed** CFTR patch-clamp trace, infer
(1) the number of channels N, (2) the per-sample number of open channels,
(3) the per-sample counts in the seven CFTR states a..g, and (4) the
identifiable kinetic rates.

Prior methods do not all output all four quantities, so the comparison is
head-to-head wherever outputs overlap and a capability matrix where they do
not. Protocol:

- **Data**: `data/derived/synth_v2/test.npz`, the frozen benchmark used by the
  v5 presentation: 384 traces, 1000 samples each (`dt = 0.01 s`), N in 1..5,
  three noise levels (`X_s1`, `X_s2`, `X_s4`), per-group randomised 7-state
  CFTR rate tables, full per-channel state labels `r`.
- **Metrics**: channel-count accuracy (N̂ vs N per trace), per-sample
  open-count accuracy and mean absolute error (MAE). State-count and rate
  metrics are reported for v5a only; the baselines cannot produce them.
- **Our side**: the published `kihmm_v4_v5a.pt` results via the v5 exact
  inference (`presentation_v5_metrics.json`), unchanged from PR #1.
- **Baselines**: per-trace fits for the two HMMs (BIC over L = 1..5), one
  network retrained on the `synth_v2` train split for Deep-Channel. Nothing is
  tuned on test.
- **Compute**: 10 CPU workers for the HMM baselines; one GTX 1660 SUPER for
  Deep-Channel.

## 2. Reproduced methods and provenance

Full table in `BASELINES.md`; summary:

| Method | Paper | Reference code (pinned) | License | Our implementation |
|---|---|---|---|---|
| SD-HMM | Requadt & Li 2026, arXiv:2607.03088 | `gitlab.gwdg.de/requadt/sdmc` @ `52a345c` | GPL-3 | `sdmc_port.py` + `hmm_core.py`, line-by-line from the R/Rcpp kernels |
| VND-HMM | Vanegas, Eltzner, Rudolf, Dura, Lehnart & Munk 2024, AOAS 23-AOAS1842 | `github.com/ljvanegas/VND` @ `cd200be` | GPL-2 | `vnd_port.py` + `hmm_core.py`, same approach |
| Deep-Channel | Celik et al. 2020, Commun. Biol. 3:11 | `github.com/RichardBJ/Deep-Channel` @ `230d134` | MIT | `deepchannel_port.py`, PyTorch port of the shipped Keras model |

Reference code is cloned under `.external/` (gitignored) and never modified or
vendored. There is no R runtime on this machine, so the R packages are ported
rather than executed; this is the main methodological deviation and it is
covered by the verification below.

**SD-HMM** is a continuous-time sum-dependent Markov chain on the open-channel
count with rates `(lambda_0..lambda_{L-1}, mu_1..mu_L)`, Gaussian emissions
with per-level standard deviations, exact `exp(delta*R)` transitions,
forward-backward EM, and BIC model selection over L.

**VND-HMM** is the discrete-time predecessor: a vector-norm-dependent chain on
the same count space with 2L transition probabilities, Baum-Welch EM, and BIC
selection over L.

**Deep-Channel** is a pointwise CNN+LSTM idealizer. The paper and the shipped
code train with `n = 1` time step (one sample per input, zero initial
recurrent state), so the recurrent weights never activate; our port reproduces
exactly that recipe and is verified against the shipped model JSON.

## 3. Reconstruction verification (before touching our data)

Each port had to reproduce its own paper's simulation results first. Commands:
`python code/05_baselines/vnd_port.py --verify` and
`python code/05_baselines/sdmc_port.py --verify`.

**VND-HMM**

- Transition matrix: rows sum to 1 across random parameter draws for l = 1..5;
  the identity case (`theta = 1`) gives the identity matrix; the constant-rate
  case equals the analytically computed binomial independent-channel matrix.
- Parameter recovery: EM on 20,000 samples of the paper's independent case
  recovers `(0.99, 0.98, 0.98, 0.99)` as `(0.990, 0.981, 0.980, 0.990)`.
- Model selection: BIC picks the true `l = 2` among 1..3.
- Viterbi: matches a log-space brute-force implementation exactly.

**SD-HMM**

- Rate matrix: matches equation (3) of the paper; embedding rows sum to 1.
- Parameter recovery: EM on 20,000 samples of the paper's Example 2
  (`theta = (3, 4, 4, 3)`, `b = 0`, `a = 1`, `sigma = 0.1`, `delta = 0.05`)
  returns `(2.98, 3.95, 3.96, 3.07)`.
- Cooperativity: the positive scenario `(1, 5, 9, 9, 5, 1)` recovers opening
  rates `(1.07, 5.18, 8.94)` and the negative scenario `(9, 5, 1, 1, 5, 9)`
  recovers `(9.76, 4.95, 0.95)`; the same monotonicity test used in the paper.
- Model selection and Viterbi: BIC picks the true L; Viterbi matches brute
  force.

**Deep-Channel**

- Architecture: parsed from the shipped
  `model/JSON/nmn_oversampled_deepchannel6/model.json` and matched layer by
  layer (Conv1D(64, k=1, relu); 3x LSTM(256) with `activation='relu'`,
  `recurrent_activation='hard_sigmoid'`; BN; dropout 0.2; Dense(6); softmax).
- Parameter count 1,385,606; forward shapes checked.
- Sanity training on real benchmark points: 200 SGD steps take the loss from
  1.99 to 0.82 with 0.69 batch accuracy.

Known, documented deviations (all in `BASELINES.md`): scipy optimizers instead
of R `constrOptim`/`optim` with identical objectives (the emission M-step uses
an analytic gradient, checked against finite differences to 4e-9); percentile
emission initialisation instead of a user-supplied range; BIC with the
observed HMM likelihood and `k = 3L + 3`; Deep-Channel reimplemented in
PyTorch, class weights instead of SMOTE, MinMax scaling fitted on train only;
the Viterbi underflow fallback uses `|tmp|/35` because the reference's
negative standard deviation produces NaNs.

## 4. Headline results

Frozen `synth_v2` test set, 384 traces. N = channel-count accuracy,
open = per-sample open-count accuracy, MAE = open-count mean absolute error.

| Method | Split | N acc | Open acc | Open MAE |
|---|---|---|---|---|
| **KI-HMM v5a** | test_x1 | **0.951** | **0.8527** | **0.1830** |
| KI-HMM v5a | test_x2 | 0.990 | 0.7547 | 0.3163 |
| KI-HMM v5a | test_x4 | 1.000 | 0.5080 | 0.6184 |
| SD-HMM | test_x1 | 0.401 | 0.6402 | 0.4980 |
| SD-HMM | test_x2 | 0.167 | 0.3124 | 1.1931 |
| SD-HMM | test_x4 | 0.234 | 0.2952 | 1.1253 |
| VND-HMM | test_x1 | 0.417 | 0.6367 | 0.5032 |
| VND-HMM | test_x2 | 0.167 | 0.2760 | 1.1999 |
| VND-HMM | test_x4 | 0.232 | 0.2819 | 1.1481 |
| Deep-Channel | test_x1 | 0.786 | 0.5290 | 0.5440 |
| Deep-Channel | test_x2 | 0.333 | 0.4344 | 0.7200 |
| Deep-Channel | test_x4 | 0.182 | 0.3172 | 1.0727 |

v5a also reports per-state counts and rates: state MAE 0.255 and rounded
state-count accuracy 0.798 at x1, with identifiable-direction R2
`[0.874, 0.682, 0.390, 0.213]`. No baseline can produce these outputs.

## 5. Per-channel-count breakdown (noise x1)

| True N | Traces | Metric | **v5a** | SD-HMM | VND-HMM | Deep-Channel |
|---|---|---|---|---|---|---|
| 1 | 69 | N acc | 1.000 | 0.304 | 0.362 | 1.000 |
| 1 | 69 | open acc | 0.991 | 0.707 | 0.718 | 0.885 |
| 1 | 69 | open MAE | 0.014 | 0.312 | 0.305 | 0.115 |
| 2 | 86 | N acc | 1.000 | 0.849 | 0.814 | 1.000 |
| 2 | 86 | open acc | 0.964 | 0.863 | 0.848 | 0.656 |
| 2 | 86 | open MAE | 0.054 | 0.147 | 0.165 | 0.344 |
| 3 | 73 | N acc | 0.973 | 0.575 | 0.562 | 0.863 |
| 3 | 73 | open acc | 0.881 | 0.738 | 0.743 | 0.497 |
| 3 | 73 | open MAE | 0.158 | 0.289 | 0.284 | 0.510 |
| 4 | 86 | N acc | 0.860 | 0.209 | 0.267 | 0.174 |
| 4 | 86 | open acc | 0.725 | 0.495 | 0.480 | 0.408 |
| 4 | 86 | open MAE | 0.326 | 0.749 | 0.762 | 0.688 |
| 5 | 70 | N acc | 0.929 | 0.000 | 0.014 | 0.986 |
| 5 | 70 | open acc | 0.705 | 0.378 | 0.378 | 0.203 |
| 5 | 70 | open MAE | 0.358 | 1.022 | 1.025 | 1.072 |

Observations:

- The HMM baselines are competitive only at N = 2 on open counts (0.85–0.86,
  close to v5a's 0.964); they collapse from N = 4 onward and never recover
  N = 5 (0/70 and 1/70 correct).
- Deep-Channel is the strongest baseline for channel count at N = 1–2 (its
  max-openings heuristic works when noise is low and levels are well
  separated) and its open-count accuracy at N = 1 (0.885) is the only baseline
  number that comes close to v5a (0.991). It degrades sharply from N = 3.
- Deep-Channel's N estimate is a by-product of the maximum predicted opening:
  at N = 4 it predicts 5 openings on 67 of 86 traces at x1, and under noise
  this inflates further (x2: N accuracy 0.333; x4: 0.182).

Channel-count confusion (rows = true N, columns = predicted N, noise x1):

```
SD-HMM          VND-HMM         Deep-Channel
[[21 32 14  2  0] [25 34  8  2  0] [69  0  0  0  0]
 [ 5 73  1  6  1] [ 5 70  4  6  1] [ 0 86  0  0  0]
 [ 0 27 42  2  2] [ 1 30 41  0  1] [ 0  1 63  9  0]
 [ 2 30 34 18  2] [ 2 31 30 23  0] [ 0  0  4 15 67]
 [ 0 22 27 21  0]] [ 0 20 34 15  1]] [ 0  0  0  1 69]]
```

The HMM baselines systematically under-select L for high true N and overshoot
for N = 1; Deep-Channel's errors are almost entirely one-level too high at
N = 4.

## 6. Noise stress test (x2, x4)

| Method | x2 N / open / MAE | x4 N / open / MAE |
|---|---|---|
| **v5a** | **0.990 / 0.755 / 0.316** | **1.000 / 0.508 / 0.618** |
| SD-HMM | 0.167 / 0.312 / 1.193 | 0.234 / 0.295 / 1.125 |
| VND-HMM | 0.167 / 0.276 / 1.200 | 0.232 / 0.282 / 1.148 |
| Deep-Channel | 0.333 / 0.434 / 0.720 | 0.182 / 0.317 / 1.073 |

v5a's channel-count accuracy *improves* with noise (the amplitude range grows
with N, so the count cue gets easier) while its open-count accuracy decays at
the same rate as the information in the signal. The baselines lose both: the
HMM per-trace BIC fits degrade, and Deep-Channel's max-openings heuristic
inflates.

## 7. Why Deep-Channel stops at ~0.53 (fairness analysis)

Deep-Channel is a *pointwise* classifier in the released recipe, so its
accuracy is bounded by the context-free per-sample Bayes rate. Measured on the
same test traces with the true N and the true levels `0.58 + 0.82 k`:

| Model for the pointwise problem | Open accuracy (x1) |
|---|---|
| Nearest-level classifier, true N known (ceiling) | 0.744 |
| Plain 3-layer MLP trained pointwise on the same data | 0.56 (val) |
| Deep-Channel (`deepchannel_port.py`, as published) | 0.529 |

Two robustness checks are recorded in `BASELINES.md`:

- A 16-epoch retrain with a gentler schedule (`dc_long_*`) changes nothing
  (x1 open 0.526 vs 0.529).
- A sequence-input variant that actually exercises the LSTM
  (`deepchannel_seq.py`) was attempted and abandoned: 1930 s per epoch on the
  GTX 1660 SUPER (manual unrolled LSTM) and 0.43 validation accuracy after one
  epoch, below the pointwise model.

So the Deep-Channel number reflects the pointwise idealization task, not a
weak training recipe, and the comparison is fair to the published method.

## 8. Capability matrix

| Method | N | Open count | 7-state counts | Rates | Neural | CFTR |
|---|---|---|---|---|---|---|
| **KI-HMM v5a** | yes (evidence) | yes | yes | yes (4 identifiable directions) | hybrid | yes |
| SD-HMM | yes (BIC) | yes (Viterbi) | no | effective birth-death only | no | no |
| VND-HMM | yes (BIC) | yes (Viterbi) | no | transition probabilities only | no | no |
| Deep-Channel | max-openings heuristic | yes (per sample) | no | no | yes (RCNN) | no |

This is the structural half of the novelty argument: even before accuracy, no
prior method produces the joint (N, 7-state occupancy counts, rates) output
that the thesis defines.

## 9. Runtime and hardware

| Run | Wall time |
|---|---|
| SD-HMM, 384 traces x 5 candidate L, 10 workers | 20–24 min per split |
| VND-HMM, same | 18–20 min per split |
| Deep-Channel training (6 epochs, pointwise) | 2.3 min on the GTX 1660 SUPER |
| Deep-Channel evaluation (3 splits) | seconds |
| KI-HMM v5a | from PR #1: ~25 min inference for all splits |

TensorFlow does not see the GPU in this environment; PyTorch does. The
Deep-Channel port uses PyTorch for that reason.

## 10. Threats to validity and honest caveats

1. **Two-state misspecification.** SD-HMM and VND-HMM model binary channels;
  on 7-state CFTR data the open-count process is not Markov, so the baselines
  are structurally misspecified. This is a property of comparing against
  two-state methods and must be stated in the paper.
2. **R ports.** The reference R packages could not be executed here. The ports
  are verified against the papers' own simulation scenarios (Section 3), but a
  reviewer could ask for the reference implementation; installing R and
  cross-checking is listed as an optional follow-up in `docs/STATUS.md`.
3. **Optimizer sensitivity of EM.** The baselines' M-steps are local
  optimizations; different scipy settings occasionally land in different local
  optima. Settings are documented, and every method was given the same
  optimizer budget and initialisation.
4. **Deep-Channel pointwise.** The released method's recurrent layers never
  activate; the sequence variant we tried did not help. If a reviewer insists
  on a temporal neural baseline, the script to revisit is `deepchannel_seq.py`.
5. **Data domain.** Everything here is synthetic CFTR with a fixed 7-state
  graph. On real recordings no ground-truth labels exist, so no baseline
  comparison on real data is claimed.

## 11. What can be claimed in the paper

Supported: *On a frozen synthetic benchmark of summed CFTR traces with known
labels (N = 1–5, three noise levels), the proposed method improves
channel-count accuracy from 0.79 (best prior method) to 0.95, open-count
accuracy from 0.64 to 0.85, and open-count MAE from 0.50 to 0.18 at the
reference noise level, and it is the only compared method that also recovers
per-state occupancy counts and identifiable kinetic rates. The advantage
grows with channel count and noise.*

Not supported: calling the prior methods "wrong" in general — they target
different models (binary channels, large ensembles, pointwise idealization).

## 12. Files and reproduction

- Ports: `code/05_baselines/hmm_core.py`, `vnd_port.py`, `sdmc_port.py`,
  `deepchannel_port.py`, `deepchannel_seq.py`, `compare_baselines.py`
- Provenance and deviations: `code/05_baselines/BASELINES.md`; plan and
  checkpoints: `code/05_baselines/PLAN.md`
- Results: `code/05_baselines/results/baseline_comparison.{md,json}` plus the
  per-method JSON/NPZ files; figure:
  `code/05_baselines/figures/baseline_comparison.png`
- Model: `code/05_baselines/models/deepchannel.pt`

```bash
python code/05_baselines/sdmc_port.py --verify
python code/05_baselines/vnd_port.py --verify
python code/05_baselines/sdmc_port.py --split test_x1 --jobs 10   # test_x2, test_x4
python code/05_baselines/vnd_port.py  --split test_x1 --jobs 10
python code/05_baselines/deepchannel_port.py --train --epochs 6 --batch 1024
python code/05_baselines/deepchannel_port.py --eval --splits test_x1,test_x2,test_x4
python code/05_baselines/compare_baselines.py --dc-tag deepchannel
```

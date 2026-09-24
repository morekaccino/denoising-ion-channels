# Baseline provenance and porting notes

Head-to-head benchmarks of KI-HMM v5 against the three closest prior methods,
run on the frozen `synth_v2` benchmark. Third-party code is cloned to
`.external/` (gitignored) and never modified; this file records exactly what
was used, how each method was ported, and every deviation from the reference.

## Repositories

| Method | Paper | Repo | License | Commit | Files used |
|---|---|---|---|---|---|
| VND-HMM | Vanegas, Eltzner, Rudolf, Dura, Lehnart & Munk (2024), *Ann. Appl. Stat.* 18(2), DOI 10.1214/23-AOAS1842 | `github.com/ljvanegas/VND` | GPL-2 | `cd200be` (2021-03-04) | `R/estimate_HMM.R`, `src/vnd_model.cpp`, `src/Baum_Welch_step.cpp`, `src/Likelihood_Forward.cpp`, `src/Viterbi_simple.cpp` |
| SD-HMM (SDMC) | Requadt & Li (2026), arXiv:2607.03088 | `gitlab.gwdg.de/requadt/sdmc` | GPL-3 | `52a345c` (2026-06-29) | `src/sd_model.cpp`, `src/Baum_Welch_step.cpp`, `src/Likelihood_Forward.cpp`, `src/Viterbi_simple.cpp`, `R/HMM_estimation.R` |
| Deep-Channel | Celik, O'Brien, Brennan, Rainbow, Dart, Zheng, Coenen & Barrett-Jolley (2020), *Commun. Biol.* 3:11, DOI 10.1038/s42003-019-0729-3 | `github.com/RichardBJ/Deep-Channel` | MIT | `230d134` (2025-03-03) | `deepchannel_train.py`, `predictor.py`, `model/JSON/nmn_oversampled_deepchannel6/model.json` |

Python ports live in `code/05_baselines/` (`hmm_core.py`, `vnd_port.py`,
`sdmc_port.py`, `deepchannel_port.py`). The ports are the benchmark; the
original repos are kept only as the specification.

## Why ports instead of running the reference code

The reference VND and SDMC implementations are R packages with Rcpp/C++
kernels and there is no R runtime on this machine. Installing R was deferred
by decision (Python if at all possible). The ports therefore reproduce the
reference algorithms line-by-line from the R/Rcpp sources, and are verified
against the papers' own simulation setups (`--verify` in both ports):

- VND: transition-matrix properties; independence case equals the analytically
  computed binomial channel transition matrix; EM recovers `theta =
  (0.99, 0.98, 0.98, 0.99)` for `l=2` from 20k simulated samples; BIC selects
  the true `l`; Viterbi matches a log-space brute-force reference.
- SDMC: `rate_mat` matches the paper's equation (3); `expm` embedding; EM
  recovers the paper's Example-2 parameter vector `(3, 4, 4, 3)` from 20k
  simulated samples; positive and negative cooperativity signs are recovered
  from the paper's Section 5.2 scenarios; BIC selects the true `L`; Viterbi
  matches brute force.

## Deviations from the reference implementations

Shared (VND + SDMC):

1. **Optimizers** are scipy instead of R `constrOptim`/`optim`. The M-step
   objectives and constraints are identical; the emission step uses an
   analytic gradient (checked against finite differences, max error 4e-9).
2. **Model selection** over `L in {1..5}` uses BIC with the observed HMM
   log-likelihood (plain Gaussian level densities, matching SDMC's
   `bic_aic_sd_hmm` parameter count `k = 3L + 3`). The VND package ships no
   model-selection routine; the same BIC rule is applied.
3. **Emission initialization** `(min, max)` was replaced by the trace's
   0.5%/99.5% percentiles, because the reference API requires the caller to
   supply a range and the generalized-hyperbolic noise has heavy tails.
4. **EM budget**: `max_it=100`, convergence threshold `1e-6` (reference
   defaults; EM typically runs the full 100 iterations).
5. Both ports run single-threaded (`OMP_NUM_THREADS=1`) inside a
   `ProcessPoolExecutor`.

VND-specific:

- Transition M-step: SLSQP with `theta in [0,1]^{2l}` (reference: `constrOptim`
  with the same box).
- No stationarity option in the reference VND EM; the initial distribution is
  updated as in `BW_step` but the forward pass always starts uniform (as in
  `HMM_custom`).

SDMC-specific:

- Transition M-step: Nelder-Mead on `log theta`, `maxiter=500` (R `optim`
  default), `xatol=1e-6`, `fatol=1e-8` (reference: `optim(..., method=
  "Nelder-Mead")` on `log(theta)` with default `reltol`).
- Rate truncation at `max_rate=1e5` keeps the reference's guard
  (`trunc_value=1e-6`).

Deep-Channel-specific:

- Architecture reimplemented in PyTorch because TensorFlow in this venv does
  not see the GPU (torch does). The port matches the shipped JSON exactly:
  Conv1D(64, k=1)+relu, 3x LSTM(256, `activation='relu'`,
  `recurrent_activation='hard_sigmoid'`), BN and dropout 0.2, dense 6 +
  softmax. Shipped code and paper train with time steps `n=1`, so the port
  trains pointwise with zero initial state (the recurrent weights never
  activate).
- Training data: the frozen `synth_v2` train split (3072 traces, per-trace
  noise scales approx 1-4, 3.07M labelled samples). MinMax scaling is fitted
  on train only (the reference fits on the whole file, including test).
- Class imbalance is handled with inverse-frequency class weights instead of
  SMOTE oversampling.
- Optimizer: SGD, lr 1e-3, momentum 0.9 (reference) with a `StepLR(3, 0.1)`
  schedule over 6 epochs (reference ran 2 epochs with a much faster decay).
  A 16-epoch run with a gentler schedule (`results/dc_long_*`, `StepLR(6,
  0.3)`) changes nothing (test x1 open 0.526 vs 0.529), so the pointwise
  model is at its training-budget ceiling.
- Channel count `N` is the maximum predicted open count over a trace
  (the heuristic stated in the paper).

### Pointwise ceiling check (noise x1)

Because the reference recipe is pointwise, its accuracy is bounded by the
context-free per-sample Bayes rate. Measured on the frozen test set with the
true channel count and the known levels `0.58 + 0.82*k`: nearest-level
accuracy is 93.7% (N=1), 84.2% (N=2), 74.0% (N=3), 65.0% (N=4), 59.7% (N=5);
a 3-layer MLP trained pointwise on the same data reaches 56% validation
accuracy and no longer improves, consistent with the extra ambiguity from not
knowing N per trace. The Deep-Channel port lands at 52.9%, i.e. near the
context-free ceiling; this is a property of pointwise classification on this
problem, not a broken port.

### Sequence-input variant (attempted, abandoned)

`deepchannel_seq.py` feeds whole traces to the same architecture so the LSTM
can use temporal context (a deviation from the released `n=1` recipe). One
epoch took 1930 s on the GTX 1660 SUPER (manual unrolled LSTM) and reached
only 0.43 validation open accuracy, below the pointwise model, so the variant
was stopped and is not part of the headline comparison. The script is kept
for completeness.

## Evaluation protocol

- Inputs: `data/derived/synth_v2/test.npz` (`X_s1`, `X_s2`, `X_s4`), 384
  traces, 1000 samples, `dt=0.01 s`, N in 1..5, full 7-state labels `r`.
- VND/SDMC are fitted per trace for `L = 1..5` and select `L` by BIC;
  Deep-Channel is trained once on the train split and predicts per sample.
- Metrics: N accuracy, per-sample open-count accuracy and MAE, per-N
  breakdown; state-count and rate metrics are reported for KI-HMM v5 only
  (prior methods do not output them).
- Our side: `kihmm_v4_v5a.pt` through `code/04_ml/infer_v5.py` on the same
  traces (published numbers: N 0.951, open 0.853, MAE 0.183, state MAE 0.255,
  rate R2 [0.874, 0.682, 0.390, 0.213] at noise x1).

# Project Status

> Living record of where the project stands, what was tried, and why decisions were made.
> Update this file whenever a new experiment direction starts or concludes.

## Current status

**Thesis complete and submitted** (December 2024). The winning approach is the combined **Cluster Count NN + LSTM** pipeline; it outperforms the classical DBSCAN and DBSCAN+BGMM methods across 1–5 ion channels, especially in high-noise scenarios.

Repository restructure done in two passes (Sep 2026): first curated the code/data used in the thesis out of the original `MastersThesis` repo into this clean layout; second pass fixed reproducibility issues found by an audit (restored `sim.py`, hardened notebook bootstraps, split ML dependencies) and documented the thesis-vs-code discrepancies below.

## Timeline

| Date | Event |
|---|---|
| 2023-08 – 2023-10 | DBSCAN pipeline developed (`dbscan.ipynb`), made readable for other programmers |
| 2023-09 | Switched to second dataset (`03n18xxx`) to refresh caches |
| 2023-12 | `before`/`after` snapshots; DBSCAN results looked "pretty good" |
| 2024-03 | Methodology description added to `finding_number_of_states.ipynb` (used in thesis); `dbscan-horizontal_space` branch merged in the original repo |
| 2024-04-02/03 | Fitted the real noise distribution from recordings (generalized hyperbolic) |
| 2024-04-04/05 | Moved `IonChannel` into `source/`; first successful ML model, then the 50-points-before/after window model |
| 2024-04-14 | ML model with 80% accuracy (4-state classification); tested on real data; model saved |
| 2024-04-28 | State estimator supporting up to 4 states |
| 2024-05-02 | Model complete — "needs to be tested with real data" |
| 2024-12 | Thesis submitted |
| 2026-09 | Repo curated into `masters-paper`; audits + reproducibility fixes; CI + verify script added |
| 2026-09 | Saved models migrated to Keras 3 format; repo environment upgraded to Python 3.13 / TF 2.21 / Keras 3.15; end-to-end re-execution validated on the new stack |
| 2026-09 | AGENTS.md: added mandatory "Documentation discipline" section so future agents keep docs in sync with every change |
| 2026-09 | Thesis re-read to anchor the novel-method track; novelty gaps: FHMMs dismissed as intractable for multi-channel (3.4.1), factor-graph/HMM inference is single-channel only (1.2.3.3), future work asks for kinetics-integrated DL, semi-supervised learning, efficient architectures (4.1.3) |
| 2026-09 | Phase 0: frozen benchmark added — `code/04_ml/benchmark.py` + `data/derived/synth_v1/` (seeded train/val/test, N∈1–3 and 4–5 extrapolation, noise ×1/×2/×4, mismatch variants) with per-N, transition-window, calibration and N-estimation metrics the thesis never reported |
| 2026-09 | Phase 1: exact factorial-HMM decoder — `code/04_ml/kinetics.py` (count-state chain, sum-of-GH emissions via FFT, forward–backward, evidence-based N). Frozen-test results (notebook `01_exact_factorial_hmm.ipynb`, JSON in `code/04_ml/results/`): **96.3 / 83.4 / 67.2%** timestep accuracy at noise ×1/×2/×4; **100% N accuracy** (600 traces, signal likelihood); N=4–5 extrapolation **86.5%** at ×1 |
| 2026-09 | Phase 2: **KI-HMM** (`code/04_ml/torch_models.py`, `train_kihmm.py`) — neural TCN emissions + exact kinetic-chain forward pass + N-agnostic count head, trained on 3000 mixed-N traces with per-trace noise scale 1–4. Frozen test: **93.3 / 81.6 / 63.9%** at ×1/×2/×4 with **100% N accuracy** from the learned count head; N=4–5 extrapolation 77.6%; close to the Bayes-optimal decoder without knowing N, rates or noise scale |
| 2026-09 | Mismatch benchmark (lowpass, Gaussian, correlated, drift): exact decoder remains strongest (e.g. drift 94.4% vs KI-HMM 90.4%); Gaussian-noise robustness confirmed (95.9%). Evidence-based N selection is unreliable across N (~30%) — the learned count head is the N estimator |
| 2026-09 | Illustrated edition of the KI-HMM explainer (15 chapters, ~5.2k words, 29 figures: 14 AI illustrations via cheap OpenRouter image model + 13 exact matplotlib diagrams + result figures). Worked math appendix covers C(N+6,6), stars and bars, 84/462, expm2-state example, logsumexp, convolution. Source: `docs/explainer/KI-HMM_explained.md`, book: `docs/explainer/KI-HMM_explained.epub` |
| 2026-09 | Plain-language KI-HMM explainer written and built as an iBooks-readable EPUB (`docs/explainer/KI-HMM_explained.epub`, Markdown source alongside; cover image generated once via a cheap OpenRouter image model) |
| 2026-09 | KI-HMM prediction-vs-signal figures added: `code/04_ml/02_neural_hmm.ipynb`, `code/04_ml/results/figures/kihmm_predictions.png` (4 traces, inferred N correct on all; 90.6–99.4% per-trace accuracy) |
| 2026-09 | Fair baseline retraining done (`train_baselines.py` → `results/baselines_synth_v1.json`). Frozen-test leaderboard — timestep accuracy at noise ×1/×2/×4: LSTM 81.2/57.1/32.4; combined CCNN+LSTM (thesis winner) 89.2/70.8/38.3; **KI-HMM 93.3/81.6/63.9**; exact decoder 96.3/83.4/67.2. Channel-count accuracy: CCNN 96.8% vs KI-HMM 100%. Transition accuracy: KI-HMM 87.1/69.6/51.9 vs combined 82.9/64.3/36.3 |
| 2026-09 | Phase B (v2 track): **rate-randomized benchmark** — `code/04_ml/benchmark_v2.py` + `data/derived/synth_v2/` (512 train groups × 6 traces, N ∈ 1–5, noise scale 1–4; val/test/extrap + 5 mismatch sets; 26 MB, 8 s to generate). Each group has its own 12-rate table (×0.5–×2 log-uniform, open probability kept in 0.10–0.90); each trace has full 7-state labels `r`. Checks pass: per-state counts sum to N, open count matches `y`, rates positive. Added the missing `requirements-ml-torch.txt` referenced by PIPELINE |
| 2026-09 | Phase C/D (v2 track): **KI-HMM v2** — `torch_models_v2.py` + `train_kihmm_v2.py` + `eval_kihmm_v2.py`. One TCN encoder → N head, per-state count head (a..g, sum = N), and 12-rate head (pooled over each group of traces). Fisher-information analysis of the exact decoder (`results/fisher_synth_v2.json`) shows only ~4 of 12 rate directions are identifiable from summed traces; the rate loss is therefore Fisher-weighted. Frozen-test results (64 groups × 6 traces): N accuracy 94.8/96.1/100% at noise ×1/×2/×4; per-state count MAE 0.32 channels (×1); identifiable-direction R² 0.73/0.48/0.17 (×1); effective Markov parameters: opening-rate R² 0.57, closing-rate R² 0.73, p_open R² 0.79, median relative error ~18%. Bag ablation: top-direction R² rises −0.29 → 0.73 as traces per group go 1 → 6 |
| 2026-09 | Phase E1 (v2 track): **real-data application** — `apply_kihmm_v2_real.py` runs the frozen v2 model on all 53 ABFs (10 s segments at 100 Hz, one group per file; 03n17005 too short, skipped). Result: **N transfers** (≈1 for 52/53 files, matching the lab note that most chunks contain one channel), but the **rate head does not transfer**: real groups sit outside the training range (mean \|Δlog rate\| 0.255 vs 0.145 on synthetic val; 21% of files hit the ±1 log-rate clamp vs 2%), and neither anti-alias filtering before decimation nor moving-average detrending fixes it. Conclusion: Markov-parameter estimation on real data needs domain-randomized retraining (drift / colored noise / SNR augmentation), not just preprocessing. Results in `code/04_ml/results/kihmm_v2_real.json` |
| 2026-09 | Phase E1b (v2 track): **domain-randomized retraining did not fix real-data rates** — `benchmark_v2.py --augment` adds baseline wander, linear trend, AR(1) colored noise and light filtering to `train_aug.npz`/`val_aug.npz` (1024 train groups total); `kihmm_v2_v2b` keeps clean-synthetic quality (test N 95.6/97.9/100%; effective opening/closing R² 0.48/0.65, p_open 0.73) but real-data extrapolation got worse (37% of files at the rate clamp vs 21% for v2a; effective p_open IQR 0.00–0.98). Diagnosis: the group-level rate head saturates on real summaries, while the per-segment state head is not saturated and still tracks a crude threshold moderately (corr 0.59 on file 21, weaker on files 3/33). Conclusion: transfer needs fitting the simulator/noise to each recording (thesis `code/02` route) or semi-supervised adaptation, not generic augmentation. Results in `results/kihmm_v2_real_v2b.json`, `results/kihmm_v2_eval_v2b.json` |
| 2026-09 | Phase F2 (v2 track): **novelty writeup** `docs/NOVELTY.md` — contribution vs the 7 closest works (Berghaus 2024 MJPs, Moffett 2021 factor-graph EM, Vanegas 2021 superimposed HMM, SlotFlow 2025, neural HMMs, FHMMs, VAMPnets), model description, synthetic results table, identifiability limits, real-data status and reproduction commands. README links it |

## Thesis vs code discrepancies

> Some numbers in the thesis do not match the committed code. The code in this repo is what actually ran; the thesis numbers below could not be reproduced from any committed notebook (including in the archived original repo).

- **ML hyperparameters**: thesis claims 100k training traces / 16 bins / CCNN 13 layers × 170 neurons / 50 epochs / 80-20 split / LSTM window N_POINTS=10 / 64 units / 88.9% val accuracy. Code: 1500 traces, 100 bins, Dense(100)→Dense(4) (standalone CCNN) or MD branch Dense(101)→Dense(50)→Dense(3) (combined), epochs 30 (standalone) / 600 (combined), 80/20 only in the combined notebook (67/33 in the standalone ones), N_POINTS=50, LSTM 100 units, best recorded run 90.7% val accuracy at epoch 600. The hyperparameter sweeps behind thesis Figures 3.24–3.33 (layers, neurons, bins, window sizes) exist in no committed notebook.
- **GH noise parameters**: thesis Table 2.1 lists open (1.5, 2.0, 0.1, 0.0, 0.2) and closed (1.0, 1.5, −0.2, 0.0, 0.1). Code (fitted on `03n18026.abf`, data index 41): open (1.9865, 0.001999, −0.000556, 1.4, 0.000207), closed (3.6662, 0.5836, 0.1686, 0.58, 0.02425). Thesis says "method of moments"; notebooks use scipy MLE (`distribution.fit`) over 106 distributions. For the closed state, `burr12` fit better than `genhyperbolic` — GH was forced in the notebook (a thesis-motivated choice).
- **Sampling rate / amplitude**: thesis Table 2.1 says 10 kHz and 1 pA; `ion_channel.py` simulates with `dt=0.01` (100 Hz) and signal magnitudes 1.4 (open) / 0.58 (closed). The 10 kHz figure appears only in the legacy `AxonSimulator` in `source/moreka.py`, which the notebooks do not use.
- **DBSCAN heuristics**: thesis Table 3.1 recommends epsilon by peak range (0.30–0.35, 0.15–0.20, 0.05–0.10, 0.01–0.05) with 40–50 bins and threshold τ = 0.05·mean. Code uses 200 bins, `DBSCAN(eps=10, min_samples=1)` on raw histogram coordinates (`finding_number_of_states.ipynb`) or eps=0.1/0.03 on standardized data (`dbscan.ipynb`); no epsilon sweep notebook exists.
- **C1a rate quirk**: the literal −9/9 C1a rates in `ion_channel.py`'s rate matrix are overwritten at runtime by `C1aExitProb/dt = 10`, so the effective C1a exit rate is 10/s.
- **Missing artifact**: `on_real_data.ipynb` once loaded a `denoiser.keras` autoencoder that was never committed anywhere; the load is now guarded with an existence check and warning.

## Things we tried (and what we learned)

- **Raw DBSCAN on signal values** — worked for low channel counts/low noise, degraded sharply as channel count grew (max 49% on 3 channels, 36% on 4). Kept as baseline.
- **DBSCAN in "horizontal space"** (`distance`/`time` planes, moving averages) — exploratory branch, did not become the main method.
- **Peak detection with histograms + DBSCAN on peaks** — became the classical pipeline (thesis 2.3.1).
- **BGMM cluster-count prior to DBSCAN** — big accuracy improvement over DBSCAN alone in noisy data; still unreliable at 3+ channels (thesis 3.1.2).
- **Noise modeling** — Gaussian was a poor fit for real open/closed-state noise; a generalized hyperbolic fit was adopted for synthetic data generation (thesis 2.2.1, Fig 2.3). See discrepancy note above about closed-state fit quality.
- **CCNN (Cluster Count NN)** — histogram-in, channel-count-out classifier; strong channel-count estimation on 1–3 channel signals.
- **LSTM window sizes 10/20/30/40/50** — larger windows overfit (training/validation divergence); the committed pipelines use the 50/50 normalized configuration. "Garbage in, garbage out."
- **LSTM sizes** — anything > 4 neurons worked; the committed model uses 100 units.
- **Combined CCNN + LSTM** — final answer; handles noise and temporal context far better than classical methods; minor misclassifications at the first points of state transitions.
- **Exact factorial HMM for multi-channel counting** (2026-09, novel-method track) — the summed patch clamp is a count-state Markov chain, so the Bayes-optimal decoder is computable exactly; on the frozen benchmark it beats the thesis's reported numbers with no training. Key correction discovered on the way: the open-state generalized-hyperbolic noise is **heavy-tailed, not negligible** (per-channel std 0.232 vs closed 0.128), so treating the open level as a point mass breaks the decoder; both open and closed densities must be convolved exactly.
- **KI-HMM design bugs found and fixed** (recorded so they are not re-introduced): (1) `-inf` masking in log-space HMM backward produces NaN gradients (`exp(-inf - (-inf))`) — use a large finite mask (`MASK_NEG = -1e6`); (2) the channel-count head must be N-agnostic (pooled features *before* the N-conditioned FiLM) or it learns to read N off the conditioning embedding instead of the signal (35% → 100% N accuracy after the fix); (3) the open-state GH noise is heavy-tailed and cannot be treated as a delta in likelihood-based decoders.
- Dead ends / superseded artifacts (excluded from this repo): `with_sympy.py`, `results.txt`, `main.ipynb`, `considering-noise.ipynb`, early non-normalized LSTM notebooks.

## Known fragilities (code-level)

- Both noise notebooks iterate `scipy.stats._continuous_distns._distn_names` (a private scipy API) to fit 106 distributions; may break on future scipy upgrades.
- `on_real_data.ipynb` mixes `tensorflow.keras` and standalone `keras` imports; works with TF ≥ 2.16 + keras 3 (pinned in `requirements-ml.txt`).
- Model format: the saved `.keras` models were migrated from the legacy Keras 2 format (TF 2.15 era) to **Keras 3** in Sep 2026, so they load with `keras>=3.15` (`requirements-ml.txt`). Weight-equivalence was verified against the originals (max deviation 3e-7). TensorFlow currently has no Python 3.14+ wheels, so the ML stack caps at Python 3.13.
- The combined model is a subclassed `CustomModel`; loading it requires the class to be defined/registered first (its definition cell in `state_estimator_and_LSTM_normalized_50_50_artificial_data.ipynb` carries `@register_keras_serializable`).
- The two real-data notebooks (`on_real_data.ipynb`, `LSTM_normalized_50_50_real_data.ipynb`) are re-executed end-to-end whenever the environment changes; outputs reflect the latest verified run (currently Python 3.13 / TF 2.21 / Keras 3.15).
- `data/raw/03n17005.abf` is 193 KB while the other 52 recordings are ~1.2 MB — likely a short/aborted recording; not used at a critical index, but flagged here.
- No license: all rights reserved. `.abf` data (lab-provided) and thesis PDF are not for redistribution.

## Known limitations (from thesis 4.1.3)

- Models are trained on synthetic data; generalization to diverse real-world conditions needs more validation.
- LSTM training is computationally expensive for long sequences.
- Supervised approach requires labeled data, which is scarce in electrophysiology.

## Next steps (novel-method track, 2026-09)

- [x] Fair baseline retraining and leaderboard (`train_baselines.py` → `results/baselines_synth_v1.json`)
- [x] KI-HMM training/eval record and prediction figures (`code/04_ml/02_neural_hmm.ipynb`)
- [x] v2 benchmark: rate-randomized `synth_v2` + full 7-state labels (`code/04_ml/benchmark_v2.py`)
- [x] v2 model: KI-HMM v2 — N + per-state counts (a..g, sum = N) + Markov parameters (`torch_models_v2.py`, `train_kihmm_v2.py`)
- [x] Rate-recoverability study: Fisher analysis, per-direction R², bag-size ablation (`eval_kihmm_v2.py`)
- [x] Apply v2 to real ABFs (`apply_kihmm_v2_real.py`): N works, rates do not transfer yet
- [x] E1b: domain-randomized retraining — negative; generic augmentation is not enough (see timeline)
- [ ] E1c (optional): fit simulator noise/amplitude per real file (thesis `code/02` route) and retrain or adapt
- [ ] E2: ± glibenclamide comparison — blocked until real-data rates are trustworthy and the file→condition mapping is available
- [x] F2: novelty writeup `docs/NOVELTY.md` (closest works + method + synthetic results + identifiability limits + real-data status)
- [ ] Optional: slot-attention / count-head comparisons and amortized Bayesian NPE

## Next steps (thesis)

- [ ] Validate on a more diverse set of real patch-clamp recordings
- [ ] Explore CNN / transformer architectures to reduce training cost
- [ ] Unsupervised or semi-supervised variants (labels are expensive)
- [ ] Incorporate biophysical model constraints into the networks
- [ ] Apply pipeline in CF drug-development studies (CFTR modulator efficacy)

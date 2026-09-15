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
| 2026-09 | Repo curated into `masters-paper`; audits + reproducibility fixes; CI + verify script added; end-to-end re-execution validated on pinned TF 2.15.1 |

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
- Dead ends / superseded artifacts (excluded from this repo): `with_sympy.py`, `results.txt`, `main.ipynb`, `considering-noise.ipynb`, early non-normalized LSTM notebooks.

## Known fragilities (code-level)

- Both noise notebooks iterate `scipy.stats._continuous_distns._distn_names` (a private scipy API) to fit 106 distributions; may break on future scipy upgrades.
- `on_real_data.ipynb` mixes `tensorflow.keras` and standalone `keras` imports; works with TF ≥ 2.16 + keras 3 (pinned in `requirements-ml.txt`).
- The saved `.keras` models are version-sensitive: they load with **TensorFlow 2.15.1 / Keras 2.15.0** (pinned in `requirements-ml.txt`); Keras 3 refuses the legacy LSTM/initializer configs. Verified 2026-09: all three models load (the combined one needs its `CustomModel` class from its training notebook in scope).
- The two real-data notebooks (`on_real_data.ipynb`, `LSTM_normalized_50_50_real_data.ipynb`) were re-executed end-to-end on the pinned environment in Sep 2026; their outputs reflect that run.
- `data/raw/03n17005.abf` is 193 KB while the other 52 recordings are ~1.2 MB — likely a short/aborted recording; not used at a critical index, but flagged here.
- No license: all rights reserved. `.abf` data (lab-provided) and thesis PDF are not for redistribution.

## Known limitations (from thesis 4.1.3)

- Models are trained on synthetic data; generalization to diverse real-world conditions needs more validation.
- LSTM training is computationally expensive for long sequences.
- Supervised approach requires labeled data, which is scarce in electrophysiology.

## Next steps

- [ ] Validate on a more diverse set of real patch-clamp recordings
- [ ] Explore CNN / transformer architectures to reduce training cost
- [ ] Unsupervised or semi-supervised variants (labels are expensive)
- [ ] Incorporate biophysical model constraints into the networks
- [ ] Apply pipeline in CF drug-development studies (CFTR modulator efficacy)

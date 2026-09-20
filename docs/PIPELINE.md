# Pipeline: Thesis Section → Code Map

How each part of the thesis maps to the code in this repository, in execution order.

| Thesis section | What it does | Code |
|---|---|---|
| 2.1 Physical model | 7-state CFTR kinetic model, patch-clamp current summation | `source/ion_channel.py`, `source/patch_clamp.py` |
| 2.2 Data simulation | Generate synthetic traces from the model | `code/01_simulation/producing_artificial_data.ipynb` |
| 2.2.1 Noise function | Fit state-dependent generalized hyperbolic noise to real data | `code/02_noise_modeling/noise_imitation_of_real_data_open_state.ipynb`, `noise_imitation_of_real_data_close_state.ipynb` |
| 2.3.1 DBSCAN | Histogram → peak detection → DBSCAN clustering | `code/03_classical/dbscan.ipynb` |
| 2.3.2 DBSCAN + BGMM | BGMM for cluster count, DBSCAN for state assignment | `code/03_classical/finding_number_of_states.ipynb` |
| 2.4.1 Cluster Count NN | Neural net predicting number of active channels from histograms | `code/04_ml/state_estimator_normalized_50_50_artificial_data.ipynb` |
| 2.4.2 LSTM | LSTM state classification using CCNN output + time context | `code/04_ml/state_estimator_and_LSTM_normalized_50_50_artificial_data.ipynb`, `code/04_ml/LSTM_normalized_50_50_artificial_data.ipynb` |
| 3 Results | Parameter sweeps (epsilon, n_min, bins, layers, neurons, window size), head-to-head comparisons | Outputs embedded in the notebooks above; note: some sweeps described in the thesis are not present in any committed notebook (see `docs/STATUS.md`) |
| Real-data application | Classical + LSTM applied to `.abf` recordings | `code/03_classical/on_real_data.ipynb`, `code/04_ml/LSTM_normalized_50_50_real_data.ipynb` |
| — | Open-probability estimation with BGMM + sympy | `code/02_noise_modeling/normalization.ipynb` |

## Novel-method track (2026-09, in progress)

Goal: a method that has never been applied to multi-channel CFTR counting — **KI-HMM**, a kinetics-informed neural HMM that learns per-sample emissions but performs exact structured inference through the biophysical count-state chain, and infers N trans-dimensionally by marginal likelihood (the thesis dismissed FHMMs as intractable for multi-channel, and factor-graph inference is single-channel only).

v2 (stages 1b/2b) keeps the exact chain and adds a rate-randomized benchmark plus heads for the per-state counts a..g (sum = N) and the 12 Markov parameters; the rate table is what varies across groups, so estimating it is a real task. Fisher information shows only ~4 of the 12 rate directions are identifiable from summed traces; the loss weights those directions and results are reported per direction (test ×1: R² 0.73/0.48/0.17; effective opening/closing rate R² 0.57/0.73, p_open 0.79).

| Stage | What it does | Code |
|---|---|---|
| 0 — Benchmark | Seeded frozen dataset + metrics (per-N, transition-window, calibration, N) | `code/04_ml/benchmark.py`, `data/derived/synth_v1/` |
| 1 — Exact decoder | Count-state factorial HMM, exact sum-of-GH emissions, forward–backward, evidence-based N | `code/04_ml/kinetics.py`, `code/04_ml/01_exact_factorial_hmm.ipynb` |
| 2 — KI-HMM | Neural sequence encoder + exact kinetic-chain layer + N-agnostic count head; trained on mixed N and noise scales; prediction-vs-signal figures | `code/04_ml/torch_models.py`, `code/04_ml/train_kihmm.py`, `code/04_ml/02_neural_hmm.ipynb`, model `code/04_ml/models/kihmm_synth_v1.pt`, results/figures in `code/04_ml/results/` |
| 1b — Rate benchmark | Rate-randomized groups with full 7-state labels for N, per-state count and Markov-parameter targets | `code/04_ml/benchmark_v2.py`, `data/derived/synth_v2/` |
| 2b — KI-HMM v2 | Encoder + group mixer + dwell-rate branch → N head, per-state count head (a..g, sum = N), 12-rate head trained with a Fisher-weighted loss; per-direction R² and effective-rate evaluation | `code/04_ml/torch_models_v2.py`, `train_kihmm_v2.py`, `eval_kihmm_v2.py`, model `code/04_ml/models/kihmm_v2_v2a.pt`, results `code/04_ml/results/kihmm_v2_eval.json` |
| 3 — Trans-dimensional variants | Slot-attention / count-head comparisons from speech separation and NILM | (planned) |
| — Baselines | Re-train the three thesis Keras models on frozen splits for a fair comparison | `code/04_ml/train_baselines.py` (done), results `code/04_ml/results/baselines_synth_v1.json` |
| 4 — Amortized Bayesian | Neural posterior/evidence estimation, calibrated uncertainty | (planned) |
| 5 — Real data (v2) | Frozen v2 applied to all 53 ABFs (N and effective rates per file). N transfers (≈1 everywhere); rates are out-of-distribution and need domain-randomized retraining | `code/04_ml/apply_kihmm_v2_real.py`, results `code/04_ml/results/kihmm_v2_real.json` |

Plain-language explainer of the KI-HMM method (EPUB + Markdown): `docs/explainer/`.

Environment for stages 2–4: `requirements-ml-torch.txt` (torch with CUDA; TF/Keras stack untouched). TensorFlow and PyTorch both see the GTX 1660 SUPER (6 GB) after `uv pip install "tensorflow[and-cuda]" torch`.

## Saved models

`code/04_ml/models/` contains the trained Keras models:

- `state_estimator_normalized_50_50_artificial_data.keras` — Cluster Count NN
- `LSTM_normalized_50_50_artificial_data.keras` — standalone LSTM state classifier (also loaded by the real-data LSTM notebook)
- `state_estimator_and_LSTM_normalized_50_50_artificial_data.keras` — combined CCNN + LSTM pipeline

## Reproducibility notes

- Notebooks were relocated from the original `MastersThesis` repo layout into `code/<stage>/`. Each notebook starts with a bootstrap cell that computes `ROOT` (the repo root) from the notebook path, with a cwd-walk fallback, so `source/` imports and `data/`/`models/` paths resolve regardless of where the Jupyter server runs. This is the only structural change to notebook contents besides path repointing.
- Original relative references: `dirname='./data'` → `data/raw/`; `'../data'` → `data/raw/`; `'models/*.keras'` → `code/04_ml/models/*.keras`.
- `sim.py` (signal-generation helpers imported by several notebooks) lives in `source/`; notebook imports were updated from `from sim import ...` to `from source.sim import ...`.
- In `code/03_classical/on_real_data.ipynb`, the dead `import mdn` was removed and the `denoiser.keras` load (an artifact that was never committed, even in the original repo) is guarded with an existence check; the downstream predict/plot cells skip when it is absent.
- Notebooks keep their original execution outputs (the figures in the thesis were produced from them). Rerunning ML notebooks requires `tensorflow`/`keras` (`requirements-ml.txt`) and will regenerate synthetic training data.
- The saved models are in the Keras 3 format (migrated Sep 2026); the combined model needs its `CustomModel` class in scope to load (see the definition cell in its training notebook).
- Files excluded from the original repo as superseded/scratch: `with_sympy.py`, `results.txt`, `main.ipynb`, `considering-noise.ipynb`, and two early LSTM notebooks (100-point window, and non-normalized 50/50) replaced by the normalized versions kept here.

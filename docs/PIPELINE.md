# Pipeline: Thesis Section → Code Map

How each part of the thesis maps to the code in this repository, in execution order.

| Thesis section | What it does | Code |
|---|---|---|
| 2.1 Physical model | 7-state CFTR kinetic model, patch-clamp current summation | `source/ion_channel.py`, `source/patch_clamp.py` |
| 2.2 Data simulation | Generate synthetic traces from the model | `code/01_simulation/producing_artificial_data.ipynb` |
| 2.2.1 Noise function | Fit state-dependent generalized hyperbolic noise to real data | `code/02_noise_modeling/noise_imitation_of_real_data_open_state.ipynb`, `noise_imitation_of_real_data_close_state.ipynb` |
| 2.3.1 DBSCAN | Histogram → peak detection → DBSCAN clustering | `code/03_classical/dbscan.ipynb` |
| 2.3.2 DBSCAN + BGMM | BGMM for cluster count, DBSCAN for state assignment | `code/03_classical/finding_number_of_states.ipynb` |
| 2.4.1 Cluster Count NN | Neural net predicting number of active channels from histograms | `code/04_ml/state_estimator_normalized_50points_behind_50points_ahead_classification_with_artificial_data.ipynb` |
| 2.4.2 LSTM | LSTM state classification using CCNN output + time context | `code/04_ml/state_estimator_and_LSTM_normalized_50_50_artificial_data.ipynb`, `code/04_ml/LSTM_normalized_50points_behind_50points_ahead_classification_with_artificial_data.ipynb` |
| 3 Results | Parameter sweeps (epsilon, n_min, bins, layers, neurons, window size), head-to-head comparisons | Outputs embedded in the notebooks above |
| Real-data application | Classical + LSTM applied to `.abf` recordings | `code/03_classical/on_real_data.ipynb`, `code/04_ml/LSTM_normalized_50points_behind_50points_ahead_classification_with_real_data.ipynb` |
| — | Open-probability estimation with BGMM + sympy | `code/02_noise_modeling/normalization.ipynb` |

## Saved models

`code/04_ml/models/` contains the trained Keras models:

- `state_estimator_normalized_classification_with_artificial_data.keras` — Cluster Count NN (80% accuracy on 4-state classification)
- `LSTM_normalized_50points_behind_50points_ahead_classification_with_artificial_data.keras` — standalone LSTM state classifier
- `state_estimator_and_LSTM_normalized_50_50_artificial_data.keras` — combined CCNN + LSTM pipeline

## Reproducibility notes

- Notebooks were relocated from the original `MastersThesis` repo layout into `code/<stage>/`. Each notebook now starts with a bootstrap cell that computes `ROOT` (the repo root) from the notebook path, so `source/` imports and `data/`/`models/` paths resolve to absolute locations. This is the only change made to notebook contents besides path repointing.
- Original relative references: `dirname='./data'` → `data/raw/`; `'../data'` → `data/raw/`; `'models/*.keras'` → `code/04_ml/models/*.keras`.
- Notebooks keep their original execution outputs (the figures in the thesis were produced from them). Rerunning ML notebooks requires `tensorflow` and will regenerate synthetic training data.
- Files excluded from the original repo as superseded/scratch: `sim.py`, `with_sympy.py`, `results.txt`, `main.ipynb`, `considering-noise.ipynb`, and two early LSTM notebooks (100-point window, and non-normalized 50/50) replaced by the normalized versions kept here.

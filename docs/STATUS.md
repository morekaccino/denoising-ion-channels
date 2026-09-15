# Project Status

> Living record of where the project stands, what was tried, and why decisions were made.
> Update this file whenever a new experiment direction starts or concludes.

## Current status

**Thesis complete and submitted** (December 2024). All results described in the thesis are reproducible from this repository. The winning approach is the combined **Cluster Count NN + LSTM** pipeline; it outperforms the classical DBSCAN and DBSCAN+BGMM methods across 1–5 ion channels, especially in high-noise scenarios.

Last major work: repository restructure (Sep 2026) — curated the code/data actually used in the thesis out of the original `MastersThesis` repo into this clean, documented layout.

## Timeline

| Date | Event |
|---|---|
| 2023-08 – 2023-10 | DBSCAN pipeline developed (`dbscan.ipynb`), made readable for other programmers |
| 2023-09 | Switched to second dataset (`03n18xxx`) to refresh caches |
| 2023-12 | `before`/`after` snapshots; DBSCAN results looked "pretty good" |
| 2024-03 | Methodology description added to `finding_number_of_states.ipynb` (used in thesis); merged `dbscan-horizontal_space` branch |
| 2024-04-02/03 | Fitted the real noise distribution from recordings (generalized hyperbolic) |
| 2024-04-04/05 | Moved `IonChannel` into `source/`; first successful ML model, then the 50-points-before/after window model |
| 2024-04-14 | ML model with 80% accuracy (4-state classification); tested on real data; model saved |
| 2024-04-28 | State estimator supporting up to 4 states |
| 2024-05-02 | Model complete — "needs to be tested with real data" |
| 2024-12 | Thesis submitted |

## Things we tried (and what we learned)

- **Raw DBSCAN on signal values** — worked for low channel counts/low noise, degraded sharply as channel count grew (max 49% on 3 channels, 36% on 4). Kept as baseline.
- **DBSCAN in "horizontal space"** (`distance`/`time` planes, moving averages) — exploratory branch, did not become the main method.
- **Peak detection with histograms + DBSCAN on peaks** — became the classical pipeline (thesis 2.3.1).
- **BGMM cluster-count prior to DBSCAN** — big accuracy improvement over DBSCAN alone in noisy data; still unreliable at 3+ channels (thesis 3.1.2).
- **Noise modeling** — Gaussian was a poor fit for real open/closed-state noise; generalized hyperbolic fit with method-of-moments parameters was adopted for synthetic data generation (thesis 2.2.1, Fig 2.3).
- **CCNN (Cluster Count NN)** — histogram input, 13 layers × 170 neurons, 16 bins; strong channel-count estimation.
- **LSTM window sizes 10/20/30/40/50** — larger windows overfit (training/validation divergence); window of 10 points before/after was best in early experiments, final pipeline used the 50/50 normalized configuration. "Garbage in, garbage out."
- **LSTM sizes** — anything > 4 neurons worked; 64 neurons gave best validation accuracy (88.9%).
- **Combined CCNN + LSTM** — final answer; handles noise and temporal context far better than classical methods; minor misclassifications at the first points of state transitions.
- Dead ends / superseded artifacts (excluded from this repo): `sim.py`, `with_sympy.py`, `results.txt`, `main.ipynb`, `considering-noise.ipynb`, early non-normalized LSTM notebooks.

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

# Deciphering Ion Channel Dynamics: A Clustering Approach for Signal Analysis

Companion repository for the MASc thesis by **Mohammadreza Kazemi** (York University, December 2024).

The thesis investigates clustering techniques for analyzing multi-channel **CFTR ion channel activity** recorded with the patch-clamp method: classical approaches (DBSCAN, DBSCAN + Bayesian Gaussian Mixture) and machine-learning approaches (Cluster Count Neural Network + LSTM).

- Thesis: [`Kazemi_Mohammadreza_2024_MASc.pdf`](Kazemi_Mohammadreza_2024_MASc.pdf)
- Data provenance and formats: [`docs/DATA.md`](docs/DATA.md)
- Methodology and how-to-reproduce: [`docs/PIPELINE.md`](docs/PIPELINE.md)
- Project status log and experiment history: [`docs/STATUS.md`](docs/STATUS.md)

## Repository structure

```
.
├── Kazemi_Mohammadreza_2024_MASc.pdf   Thesis
├── data/
│   ├── raw/                            Raw patch-clamp recordings (.abf, 53 files)
│   └── references/                     Papers on glibenclamide block of CFTR
├── source/                             Core simulation/data-loading package
│   ├── ion_channel.py                  7-state CFTR kinetic model
│   ├── patch_clamp.py                  Multi-channel current summation
│   └── moreka.py                       ABF loading utilities (AxonData)
├── code/
│   ├── 01_simulation/                  Synthetic data generation
│   ├── 02_noise_modeling/              Noise distribution fitting on real data
│   ├── 03_classical/                   DBSCAN / DBSCAN+BGMM analyses
│   └── 04_ml/                          CCNN + LSTM notebooks and saved models
├── docs/                               Documentation
├── AGENTS.md                           Guide for AI coding agents
└── requirements.txt                    Python dependencies
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Every notebook starts with a bootstrap cell that locates the repo root, so notebooks run correctly regardless of where the Jupyter server is started. `data/` and saved models are treated as read-only inputs; new experiments belong in `code/`.

## Pipeline at a glance

1. **Simulate** CFTR currents from the 7-state kinetic model (`source/`, `code/01_simulation/`)
2. **Model noise** by fitting generalized hyperbolic distributions to real recordings (`code/02_noise_modeling/`)
3. **Cluster** with DBSCAN and DBSCAN+BGMM (`code/03_classical/`)
4. **Learn** ion-channel count (CCNN) and point-wise state (LSTM) (`code/04_ml/`)

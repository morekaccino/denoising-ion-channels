# AGENTS.md — Repository Guide for AI Agents

## Project

MASc thesis (York University, 2024): **Deciphering Ion Channel Dynamics** — clustering multi-channel CFTR patch-clamp recordings to determine the number of active ion channels and classify their states. Approaches: classical clustering (DBSCAN, DBSCAN+BGMM) and ML (Cluster Count NN + LSTM). Thesis PDF: `Kazemi_Mohammadreza_2024_MASc.pdf`.

Read `docs/STATUS.md` first — it records the current status, what was tried, known limitations, and known **thesis-vs-code discrepancies**. Keep it updated whenever experiments start or conclude.

## Documentation discipline (mandatory)

The docs are this repo's memory — future agents and the human rely on them to understand what changed and why. **Every change you make must be reflected in the docs in the same commit.** Before finishing any task:

- `docs/STATUS.md` — always update: append a dated timeline row and fix any "Things we tried" / "Known fragilities" / "Thesis vs code discrepancies" entries your change affects. If you tried something and it failed, record it before dropping it.
- `docs/PIPELINE.md` — update whenever the pipeline changes: notebook renames/moves, new notebooks, saved models, data inputs, or reproduction steps.
- `docs/DATA.md` — update for anything about the data: provenance, index→file mapping, format, rights notes.
- `README.md` — update for structure changes, quick-start changes, or new files an outsider should know about.
- `AGENTS.md` — update if conventions, commands, or key facts change. `CLAUDE.md` is just an `@AGENTS.md` import; keep it that way (single source of truth).
- After structural changes, run `python scripts/verify_repo.py` and make sure it passes.
- Never leave a doc describing something you renamed, deleted, or superseded. If unsure what changed, check `git diff --stat` and update every doc that mentions the affected paths.
- Put meaningful context in the commit message body too, so `git log` stays readable.

## Repository map

- `source/` — core Python package: `ion_channel.py` (7-state CFTR kinetic model), `patch_clamp.py` (multi-channel sum), `moreka.py` (ABF loading), `sim.py` (small signal helpers)
- `code/01_simulation/` — synthetic data generation
- `code/02_noise_modeling/` — noise distribution fitting (open/close states), open-probability estimation
- `code/03_classical/` — DBSCAN and DBSCAN+BGMM analyses
- `code/04_ml/` — CCNN + LSTM notebooks; `code/04_ml/models/` holds saved `.keras` models
- `data/raw/` — 53 raw `.abf` recordings (read-only); `data/references/` — background papers
- `docs/` — `DATA.md` (provenance), `PIPELINE.md` (thesis-section → code map), `STATUS.md` (status log), `NOVELTY.md` (v2 novelty + results)
- `scripts/verify_repo.py` — repo health smoke test (notebook validity, path resolution, counts)

## Conventions

- **Notebook bootstrap**: every notebook in `code/` starts with a cell defining `ROOT` (repo root, derived from `__file__` with a cwd-walk fallback) and appends it to `sys.path`. Always use `ROOT`-based absolute paths (`str(ROOT / 'data' / 'raw')`) instead of relative paths in new notebooks; keep the bootstrap cell as the first cell.
- **Data is read-only**: never modify or regenerate `data/raw/*.abf`. Work on copies or derived artifacts elsewhere.
- **New experiments**: create a new notebook under the appropriate `code/<NN_stage>/` folder; follow the existing numbered ordering. Saved models go in `code/04_ml/models/` with a descriptive name.
- **Documentation**: follow the mandatory "Documentation discipline" section above — docs and code changes ship together.
- Notebook outputs are tracked on purpose (thesis figures came from them); do not strip them.
- No test suite exists — run `python scripts/verify_repo.py` after structural changes, and re-run the affected notebook when feasible.

## Commands

```bash
# First-time setup (classical pipeline)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ML pipeline (code/04_ml) additionally needs tensorflow.
# IMPORTANT: the saved models are in the Keras 3 format (keras>=3.15), and
# tensorflow has no wheels for Python 3.14+ - create the venv with
# Python <= 3.13 (e.g. `uv venv --python 3.13 .venv`).
pip install -r requirements-ml.txt

# Novel-method track (KI-HMM v2/v3/v4) needs torch:
uv venv --python 3.13 .venv && uv pip install -r requirements-ml-torch.txt

# Prior-method benchmark (SD-HMM, VND-HMM, Deep-Channel, IDC, Moffett) on frozen synth_v2:
python code/05_baselines/sdmc_port.py --verify
python code/05_baselines/vnd_port.py --verify
python code/05_baselines/idc_port.py --verify
python code/05_baselines/moffett_port.py --verify
python code/05_baselines/sdmc_port.py --split test_x1 --jobs 10
python code/05_baselines/vnd_port.py --split test_x1 --jobs 10
python code/05_baselines/idc_port.py --split test_x1 --jobs 10
python code/05_baselines/moffett_port.py --split test_x1 --jobs 10
python code/05_baselines/deepchannel_port.py --train --epochs 6 --batch 1024
python code/05_baselines/deepchannel_port.py --eval
python code/05_baselines/v5_reference.py --splits test_x1   # per-trace v5 preds for subset comparisons
python code/05_baselines/compare_baselines.py --dc-tag deepchannel
python code/05_baselines/noise_sweep.py --method all --jobs 10   # 10-point noise sweep (~9 h)
python code/05_baselines/figures_noise_sweep.py
python code/05_baselines/build_gauss_splits.py                   # Gaussian-noise training splits
python code/05_baselines/assumption_controls.py --cell all --method all --jobs 10   # fairness 2x2

# Work with notebooks
jupyter notebook

# Repo health check
python scripts/verify_repo.py
```

## Key facts

- Real data: WT-CFTR recordings ± 50 µM glibenclamide (details in `docs/DATA.md`).
- The thesis's winning pipeline is the combined CCNN + LSTM (`code/04_ml/state_estimator_and_LSTM_normalized_50_50_artificial_data.ipynb`).
- The best model on the novel-method track is **KI-HMM v5a** (`code/04_ml/torch_models_v4.py`, `models/kihmm_v4_v5a.pt`) run through `code/04_ml/infer_v5.py`: one network for N, the per-state counts a..g and the 12 Markov rates, which rebuilds the kinetic chain from its own predicted rates inside the forward pass. At inference the same learned likelihood picks N by model evidence and refines the rates by gradient ascent. Trains in ~55 min on an Apple M4 Pro CPU. Each layer has a `--verify` mode that checks it against the numpy reference in `kinetics.py`.
- Before proposing a new architecture, read the bake-off results (`results/bakeoff_count.json`, `results/bakeoff_rates.json`). 23 head and backbone designs were compared and the conclusion was that architecture barely matters here; training data volume and optimiser coupling do.
- Prior-method benchmarks live in `code/05_baselines/` (Python ports of SD-HMM 2026, VND-HMM 2024, Deep-Channel 2020, IDC 2025 and Moffett 2022; R/TF/Zenodo reference code pinned under `.external/`, never vendored). Provenance, licenses and every porting deviation are in `code/05_baselines/BASELINES.md`; the head-to-head table against v5a is `code/05_baselines/results/baseline_comparison.md`; the full paper-ready write-up (protocol, verification, per-N results, fairness analysis, threats to validity) is `docs/BASELINE_BENCHMARK_REPORT.md`. Re-run the ports' `--verify` after touching `hmm_core.py` (IDC/Moffett have their own verify).
- `tensorflow` is required only for `code/04_ml/` (via `requirements-ml.txt`); classical notebooks need only the core requirements.
- The combined model is a subclassed `CustomModel`; to load it, the class must be defined/registered first (run its definition cell in `code/04_ml/state_estimator_and_LSTM_normalized_50_50_artificial_data.ipynb`, or define the same class in your script).
- Some thesis hyperparameters (ML sweeps, GH noise parameters, DBSCAN epsilon heuristics) do not match the committed notebooks — see `docs/STATUS.md` before claiming full reproducibility.
- Original history lives in the archived repo `morekaccino/MastersThesis`; this repo is the curated, active one.
- No license declared — all rights reserved; data and thesis PDF are not for redistribution.

# Plan: head-to-head benchmark vs prior methods

Approved 2026-09-23. Branch `baselines-vs-kihmm`. Decisions: Python ports
only (no R install unless unavoidable), external repos under `.external/`
(gitignored), three baselines (SDMC, VND-HMM, Deep-Channel), Deep-Channel
training budget assessed on this PC (cap only if too long).

Goal: show on the frozen `synth_v2` benchmark that KI-HMM v5 improves on the
closest prior methods. Deliver paper-ready comparison tables and figures.

## Phase 0 - setup
- [x] Branch `baselines-vs-kihmm` from `main`
- [x] `.external/` added to `.gitignore`; `code/05_baselines/` scaffold
- [x] Clone and pin baseline repos (hashes in `BASELINES.md`)

## Phase 1 - code & method acquisition
- [x] Record provenance, licenses, commit hashes (`BASELINES.md`)
- [x] Read reference implementations and map every learned/fixed step
- [x] Python ports: `hmm_core.py`, `vnd_port.py`, `sdmc_port.py`
      (R/Rcpp -> Python, line-by-line, deviations documented)
- [x] Python port: `deepchannel_port.py` (PyTorch, architecture verified
      against the shipped model JSON)

## Phase 2 - adapters & evaluation contract
- [x] `hmm_core.load_traces` for `synth_v2` (`test_x1/x2/x4`, N, y, r, R)
- [x] Per-trace emission handling for the two-state baselines
- [x] Deep-Channel training on the `synth_v2` train split (scales 1-4)

## Phase 3 - verify each reconstruction on its own terms
- [x] VND `--verify`: transition properties, independence equivalence,
      EM recovery of the paper's simulation scenario, BIC, Viterbi
- [x] SDMC `--verify`: rate matrix equation, EM recovery of Example 2
      `(3,4,4,3)`, positive/negative cooperativity recovery, BIC, Viterbi
- [x] Deep-Channel `--verify`: shipped-JSON architecture match, forward
      shapes, 200-step sanity training
- [x] Record verification outputs in `BASELINES.md` / status log

## Phase 4 - run on our synthetic data
- [x] Background launcher for SDMC + VND on `test_x1/x2/x4` (10 workers); all six runs completed (~20 min each)
- [x] SDMC full results (`results/sdmc_test_x*.json` + `_paths.npz`): N 0.401/0.167/0.234, open 0.640/0.312/0.295 at x1/x2/x4
- [x] VND full results (`results/vnd_test_x*.json` + `_paths.npz`): N 0.417/0.167/0.232, open 0.637/0.276/0.282
- [x] Deep-Channel trained (`models/deepchannel.pt`) and evaluated (`results/deepchannel_test_x*.json`): N 0.786/0.333/0.182, open 0.529/0.434/0.317; 16-epoch rerun identical (`dc_long_*`)

## Phase 5 - comparison against KI-HMM v5
- [x] `compare_baselines.py`: aggregate all results + v5 reference numbers, per-N breakdown, capability matrix, paper-ready table + figure (`results/baseline_comparison.md` / `.json`, `figures/baseline_comparison.png`)
- [x] Honest framing: two-state baselines are structurally misspecified on 7-state CFTR; Deep-Channel is pointwise and at the context-free ceiling; sequence-input variant attempted and abandoned (too slow, no gain)

## Phase 6 - docs
- [x] `docs/STATUS.md` timeline + next steps
- [x] `docs/PIPELINE.md` stage 6 entry + reproduce commands
- [x] `docs/NOVELTY.md` head-to-head section
- [x] `README.md` pointer; `scripts/verify_repo.py` pass

## Risks / fallbacks
- R not installed -> ports (chosen); if a reviewer asks for the exact
  reference, install R later and cross-check the port outputs.
- Deep-Channel TF/GPU mismatch -> PyTorch port (chosen; TF sees no GPU here).
- Runtime: transition M-step dominates; capped at the reference's optimizer
  budget and parallelised across traces; if total time is too long, cap the
  Deep-Channel epochs (baselines still run on all traces).
- Optimizer-sensitive EM local optima are a property of the baselines;
  documented rather than tuned away.

## Round 2 (2026-09): three more papers

Approved scope: IDC (Requadt et al. 2025), Moffett et al. 2022, Albertsen &
Hansen 1994, same criteria as round 1. Python only (no R install); branch
`baselines-round2` stacked on `baselines-vs-kihmm`.

- [x] Branch + provenance pins (`BASELINES.md`)
- [x] IDC pipeline port (`idc_port.py`): steps 2-3 exact, MUSCLE substituted
      by a validated rank-based segmenter (no Python MUSCLE exists)
- [x] IDC verification on the paper's noise scenarios (robustness ordering
      reproduced: IDC beats VND under Cauchy, not under Gaussian)
- [x] Moffett port (`moffett_port.py`) from their Zenodo Python code,
      vectorized and verified bit-close against their node implementation
- [x] Moffett on the 69 N=1 test traces (all noise levels)
- [x] v5 per-trace reference (`v5_reference.py`) for identical subsets/per-N
- [ ] Albertsen reconstruction (`albertsen_port.py`) - blocked on the full
      text; PMC blocks scripted PDF downloads. User action: save
      https://pmc.ncbi.nlm.nih.gov/articles/PMC1225503/pdf/biophysj00070-0031.pdf
      locally and point the maintainer at it
- [ ] Comparison tables/figure extended (`compare_baselines.py`)
- [ ] Docs + report + stacked PR

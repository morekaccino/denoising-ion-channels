# CLAUDE.md

See [AGENTS.md](AGENTS.md) for the full repository guide.

Key points:

- Research repo for the MASc thesis on clustering CFTR ion-channel patch-clamp data.
- Read `docs/STATUS.md` first and keep it updated after meaningful work.
- Notebooks in `code/` have a bootstrap cell defining `ROOT`; use `ROOT`-based absolute paths, never relative ones.
- `data/raw/` is read-only (53 real `.abf` recordings).
- New experiments → new notebook under `code/<NN_stage>/`; saved models → `code/04_ml/models/`.
- Setup: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`, then `jupyter notebook`.
- No test suite; validate by re-running affected notebooks.

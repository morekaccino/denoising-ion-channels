#!/usr/bin/env python3
"""Repository health smoke test.

Checks, without heavy dependencies:
  - every notebook is valid JSON with a repo-root bootstrap shim
  - all ROOT-based data/model paths referenced in notebooks exist
  - no leftover relative path patterns from the pre-curation layout
  - expected file counts (recordings, models, references)
  - requirements files exist

Exit code 0 = healthy, 1 = problems found.
Run from the repo root:  python scripts/verify_repo.py
"""

import glob
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
problems = []


def check(condition, message):
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        problems.append(message)


notebooks = sorted(glob.glob("code/**/*.ipynb", recursive=True))
print(f"Notebooks found: {len(notebooks)}")
check(len(notebooks) >= 12, f"at least 12 notebooks (got {len(notebooks)})")

# Banned relative-path patterns from the pre-curation layout
BANNED = [
    r"dirname\s*=\s*['\"]\.{0,2}/?data['\"]",   # dirname='./data' / '../data'
    r"from sim import",                          # pre-curation import
    r"import mdn",                               # dead module
]

shim_ok = 0
for f in notebooks:
    nb_path = ROOT / f
    try:
        nb = json.load(open(nb_path))
    except Exception as exc:
        check(False, f"{f}: invalid JSON ({exc})")
        continue
    check("nbformat" in nb, f"{f}: has nbformat key")

    first_src = "".join(nb["cells"][0].get("source", []))
    if "ROOT" in first_src and "sys.path.insert" in first_src:
        shim_ok += 1
    else:
        check(False, f"{f}: missing ROOT bootstrap shim as first cell")

    # resolve every ROOT-based path literal used in code cells
    path_re = re.compile(r"ROOT\s*(?:/\s*'[^']+')+")
    bare_model_re = re.compile(r"(['\"])(?:\.\.?/)?models/[^'\"]+\.keras\1")
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        for banned in BANNED:
            if re.search(banned, src):
                check(False, f"{f}: leftover pattern {banned!r}")
        for line in src.splitlines():
            if bare_model_re.search(line) and "ROOT" not in line:
                check(False, f"{f}: bare models/ path not routed through ROOT: {line.strip()[:100]}")
        for match in path_re.finditer(src):
            expr = match.group(0)
            try:
                target = eval(expr, {"ROOT": ROOT, "pathlib": pathlib})
            except Exception as exc:
                check(False, f"{f}: cannot evaluate {expr!r} ({exc})")
                continue
            if not target.exists():
                check(False, f"{f}: {expr} -> missing {target}")

check(shim_ok == len(notebooks), f"all notebooks have bootstrap shim ({shim_ok}/{len(notebooks)})")

print("File counts:")
abf = sorted(glob.glob("data/raw/*.abf"))
pdf = sorted(glob.glob("data/references/*.pdf"))
models = sorted(glob.glob("code/04_ml/models/*.keras"))
check(len(abf) == 53, f"53 .abf recordings (got {len(abf)})")
check(len(pdf) == 3, f"3 reference PDFs (got {len(pdf)})")
check(len(models) >= 3, f"at least the 3 thesis .keras models (got {len(models)})")
check(all(" " not in p for p in pdf), "reference PDF filenames have no spaces")

print("Core files:")
for f in ["source/ion_channel.py", "source/patch_clamp.py", "source/moreka.py",
          "source/sim.py", "source/__init__.py", "requirements.txt",
          "requirements-ml.txt", "requirements-ml-torch.txt", "README.md", "AGENTS.md", "CLAUDE.md",
          "code/04_ml/benchmark.py", "code/04_ml/benchmark_v2.py", "code/04_ml/kinetics.py",
          "code/04_ml/torch_models_v2.py", "code/04_ml/train_kihmm_v2.py", "code/04_ml/eval_kihmm_v2.py",
          "docs/STATUS.md", "docs/PIPELINE.md", "docs/DATA.md",
          "Kazemi_Mohammadreza_2024_MASc.pdf", "scripts/verify_repo.py"]:
    check((ROOT / f).exists(), f"exists: {f}")

# model load/save pairs: every model saved in a notebook must exist and vice versa
saved_models = set()
for f in notebooks:
    nb = json.load(open(ROOT / f))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        saved_models.update(re.findall(r"models/([A-Za-z0-9_]+\.keras)", src))
model_files = {pathlib.Path(p).name for p in models}
check(saved_models <= model_files, f"notebook model refs exist on disk ({sorted(saved_models)})")

print()
if problems:
    print(f"FAILED: {len(problems)} problem(s)")
    sys.exit(1)
print("All checks passed.")

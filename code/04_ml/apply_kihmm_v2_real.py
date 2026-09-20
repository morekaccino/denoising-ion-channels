"""Apply KI-HMM v2 to real ABF recordings: N, per-state counts, Markov parameters.

Each recording is split into 10 s segments (1000 samples at 100 Hz, matching the
training data) and treated as one group, so the rate head sees all segments of a
file together.

Writes ``code/04_ml/results/kihmm_v2_real.json`` and a figure.

Usage:
  python code/04_ml/apply_kihmm_v2_real.py --model code/04_ml/models/kihmm_v2_v2a.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code" / "04_ml"))

import benchmark_v2 as B2  # noqa: E402
import torch_models_v2 as T  # noqa: E402
from eval_kihmm_v2 import effective_params  # noqa: E402
from source.moreka import AxonData  # noqa: E402

RESULTS = ROOT / "code" / "04_ml" / "results"
FIGURES = RESULTS / "figures"
SEG = 1000


def file_segments(signal: np.ndarray, seg: int = SEG):
    sig = np.asarray(signal, dtype=np.float32)
    mu, sd = float(sig.mean()), float(sig.std())
    if sd > 0:
        sig = np.clip(sig, mu - 4 * sd, mu + 4 * sd)
    n = len(sig) // seg
    return sig[: n * seg].reshape(n, seg), n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "code" / "04_ml" / "models" / "kihmm_v2_v2a.pt"))
    parser.add_argument("--out", default="kihmm_v2_real")
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location=T.DEVICE)
    model = T.KIHMMv2(hidden=ckpt["args"].get("hidden", 64)).to(T.DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data = AxonData(dirname=str(ROOT / "data" / "raw"))
    files = list(data.filenames)
    all_segs, group_ids, seg_counts, skipped = [], [], [], []
    for g, f in enumerate(files):
        sig = data[g]["signal"].values[::100]
        segs, n = file_segments(sig)
        seg_counts.append(n)
        if n == 0:
            skipped.append({"index": g, "file": f, "samples_at_100hz": int(len(sig))})
            continue
        all_segs.append(segs)
        group_ids += [g] * n
    X = np.concatenate(all_segs, axis=0)
    group_ids = np.asarray(group_ids)
    print(f"{len(files)} files ({len(skipped)} too short), {len(X)} segments of {SEG}", flush=True)

    gids = np.asarray(sorted(set(int(v) for v in group_ids)))
    chunks = []
    with torch.no_grad():
        for s in range(0, len(files), 16):
            g = gids[s:s + 16]
            mask = np.isin(group_ids, g)
            xb = torch.as_tensor(X[mask], dtype=torch.float32, device=T.DEVICE)
            local = np.searchsorted(g, group_ids[mask])
            gb = torch.as_tensor(local, dtype=torch.long, device=T.DEVICE)
            p = T.predict(model, xb, gb)
            chunks.append((g, p))

    rows = []
    for g, p in chunks:
        mask = np.isin(group_ids, g)
        sub = group_ids[mask]
        counts = p["counts"].cpu().numpy()
        n_hat = p["N_hat"].cpu().numpy()
        rates = p["rates"].cpu().numpy()
        for j, gi in enumerate(g):
            m = sub == gi
            rows.append({
                "index": int(gi),
                "file": files[gi],
                "segments": int(seg_counts[gi]),
                "n_mean": float(n_hat[m].mean()),
                "n_mode": int(np.bincount(n_hat[m]).argmax()),
                "state_counts_mean": counts[m].mean(axis=(0, 2)).round(3).tolist(),
                "rates": rates[j].round(4).tolist(),
            })

    eff = effective_params(np.exp(np.asarray([r["rates"] for r in rows])))
    for r, e in zip(rows, eff):
        r["p_open"] = float(e[2])
        r["opening_rate_per_s"] = float(np.exp(e[0]))
        r["closing_rate_per_s"] = float(np.exp(e[1]))

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.out}.json").write_text(json.dumps({"files": rows, "skipped": skipped}, indent=2))
    print(f"{'file':14s} {'N':>4s} {'p_open':>7s} {'open/s':>7s} {'close/s':>8s}")
    for r in rows:
        print(f"{r['file']:14s} {r['n_mean']:4.1f} {r['p_open']:7.3f} "
              f"{r['opening_rate_per_s']:7.2f} {r['closing_rate_per_s']:8.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FIGURES.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        x = [r["opening_rate_per_s"] for r in rows]
        y = [r["closing_rate_per_s"] for r in rows]
        c = ["tab:blue" if r["file"].startswith("03n17") else "tab:orange" for r in rows]
        axes[0].scatter(x, y, c=c)
        for r in rows:
            axes[0].annotate(str(r["index"]), (r["opening_rate_per_s"], r["closing_rate_per_s"]), fontsize=6)
        axes[0].set_xlabel("opening rate (1/s)")
        axes[0].set_ylabel("closing rate (1/s)")
        axes[0].set_title("effective Markov parameters per file")

        n = [r["n_mean"] for r in rows]
        axes[1].hist(n, bins=np.arange(0.5, 6.5, 1))
        axes[1].set_xlabel("mean predicted N")
        axes[1].set_title("channels per file")
        fig.tight_layout()
        fig.savefig(FIGURES / f"{args.out}.png", dpi=130)
        print("saved", FIGURES / f"{args.out}.png", flush=True)
    except Exception as exc:
        print("figure skipped:", exc, flush=True)

    print("saved", RESULTS / f"{args.out}.json", flush=True)


if __name__ == "__main__":
    main()

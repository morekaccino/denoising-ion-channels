# Data

## Provenance

The `.abf` files in `data/raw/` are raw single-channel patch-clamp recordings of **wild-type CFTR**, provided by the lab (originally from Guiying). Guiying went back to old records and broke them into 60-second chunks; most chunks look like **1 channel present**, recorded at a high sampling rate.

Two datasets are present:

| Dataset | Files | Notes |
|---|---|---|
| Dataset 1 | `03n17000.abf` – `03n17037.abf` (38 files) | First dataset |
| Dataset 2 | `03n18023.abf` – `03n18037.abf` (15 files) | Second dataset, used to refresh caches in later analyses |

Both datasets come from one recording of WT-CFTR **in the absence and presence of 50 µM glibenclamide**. Glibenclamide enters the cytoplasmic entrance of the channel pore and its kinetics are completely different from those of normal channel closure — the difference between pore dynamics and gating dynamics is what makes the data interesting. Its action is voltage-dependent; it can enter in two ways (polar, charged) — one yields blocking times of a few ms, the other 10+ ms.

Notebooks reference recordings **by index** into the sorted file listing produced by `AxonData(dirname=...)` (see `source/moreka.py`), so the full set of files must be present to reproduce exact results.

## Format

ABF (Axon Binary Format) files. They are read with the `neo` library:

```python
import neo
axon_io = neo.io.AxonIO(filename="data/raw/03n17000.abf")
block = axon_io.read_block(lazy=False)
signal = block.segments[0].analogsignals[0]
```

or via the repo helper:

```python
from source.moreka import AxonData
data = AxonData(dirname="data/raw")
df = data[0]  # pandas DataFrame with 'time' and 'signal' columns
```

## References

Papers in `data/references/` (cited in the thesis, on glibenclamide-induced block of WT CFTR):

- Z.-R. Zhang, G. Cui, S. Zeltwanger, and N. A. McCarty, "Time-dependent Interactions of Glibenclamide with CFTR: Kinetically Complex Block of Macroscopic Currents," J Membrane Biol, vol. 201, no. 3, pp. 139–155, Nov. 2004.
- Z.-R. Zhang, S. Zeltwanger, and N. A. McCarty, "Steady-State Interactions of Glibenclamide with CFTR: Evidence for Multiple Sites in the Pore," J Membrane Biol, vol. 199, no. 1, pp. 15–28, May 2004.
- G. Cui, B. Song, H. W. Turki, and N. A. McCarty, "Differential contribution of TM6 and TM12 to the pore of CFTR identified by three sulfonylurea-based blockers," Pflugers Arch - Eur J Physiol, vol. 463, no. 3, pp. 405–418, Mar. 2012.

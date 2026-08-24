# StandardSparseSOM

A **baseline** GPU sparse self-organizing map that uses the standard NVIDIA **cuSPARSE**
library (`cusparseSpMM`) for the best-matching-unit step, in place of
[SparseBinSOM](https://github.com/mongrolwarrior/SparseSOM)'s bespoke binary /
feature-major BMU kernel. Its sole purpose is to give SparseBinSOM a **fair, same-GPU**
reference to compare against (somoclu's sparse kernel is CPU-only, which conflates
GPU-vs-CPU with specialized-vs-standard).

It is deliberately *not* part of SparseBinSOM: it's a measuring stick, not a shippable
alternative.

## What's the same, what's different

Same as SparseBinSOM (so the comparison is apples-to-apples):
- reads the same `.sbcsr` sparse-binary corpus and writes the same `.somw` feature-major
  codebook (`W[v*K + k]`), so [SparseBinEval](https://github.com/mongrolwarrior/SparseBinEval)'s
  harness and the common cosine metrics tool score both identically;
- same dense `K×V` codebook, same nearest-neuron rule (`argmax 2·x·w − ‖w‖²`), same kind of
  batch-SOM neighbourhood update on the map grid.

Different (this is what the comparison measures):
- **BMU = `cusparseSpMM`** (`scores = X·W`, X = samples×V sparse-binary CSR) + an argmax
  kernel. cuSPARSE is general-purpose, so it (a) does **not** exploit binary values — it
  multiplies stored `1.0`s — and (b) **materialises a dense scores block**, which can't be
  held whole, so the BMU is **tiled over samples**. Those are exactly the two costs
  SparseBinSOM's fused, binary-aware, on-the-fly-top-2 kernel avoids.
- the update uses plain CUDA (scatter-accumulate per BMU + a separable Gaussian blur over the
  grid), not SparseBinSOM's factorized box-blur.

Feature-major needs no special handling under cuSPARSE: `W[v*K+k]` is simply the **row-major**
layout of the `V×K` dense operand (`CUSPARSE_ORDER_ROW`).

## Build

Requires CUDA (cuSPARSE) + a CUDA-capable GPU.

```bash
cmake -S . -B build && cmake --build build -j
```

## Usage

```bash
./build/standardsparsesom CORPUS.sbcsr --rows 64 --cols 64 --epochs 33 \
    --sigma-init 32 --sigma-min 0.5 --save-weights out.somw
```

Options: `--rows/--cols` (or `--map N` for square), `--epochs`, `--sigma-init`/`--sigma-min`
(grid-cell neighbourhood radius, geometric decay), `--tile-mb` (dense scores-tile budget),
`--seed`, `--save-weights PATH`. It prints per-epoch σ + QE and the total wall time, and writes
the trained codebook as `.somw` for scoring.

Targeted at the modest map sizes used for the head-to-head (the dense scores tile and the
per-feature grid blur make very large maps memory-heavy — itself a result worth reporting).

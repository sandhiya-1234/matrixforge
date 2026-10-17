# CUDA Matrix Multiplication Optimizer

Progressively optimized CUDA SGEMM kernels, validated against a CPU reference and
benchmarked against cuBLAS.

> Status: starter. Naive kernel + benchmark harness done. Tiled and register-blocked
> kernels are next.

## Kernels

| # | Kernel | Idea | Status |
|---|--------|------|--------|
| 1 | `naive` | one thread per output element, global-memory reads | done |
| 2 | `coalesced` | indexing so neighbouring threads touch neighbouring addresses | todo |
| 3 | `tiled` | shared-memory tiles, reuse data across a block | todo |
| 4 | `register_blocked` | each thread computes several outputs in registers | todo |

## Layout

```
include/   matmul.cuh (launcher declarations), cpu_baseline.h
src/       cpu_baseline.cpp, naive.cu, main.cu (validation + benchmarking)
tests/     hello_cuda.cu (environment check)
scripts/   plot_results.py
results/   benchmark.csv, gflops.png (generated)
```

## Build and run

See [COLAB.md](COLAB.md) for the free-GPU workflow. With a local NVIDIA GPU:

```
cmake -B build && cmake --build build -j
./build/matmul_bench            # default sizes
./build/matmul_bench 512 1024   # custom square sizes
python scripts/plot_results.py
```

## Methodology

- FP32, row-major, fixed RNG seed (42).
- Timing: CUDA events, kernel only (no host<->device copies), 3 warm-up runs,
  median of 10.
- Throughput: GFLOP/s = 2*M*N*K / time.
- Correctness: max relative error vs CPU baseline (cuBLAS as reference above
  1024^3 where the CPU baseline is too slow), tolerance 1e-3. Includes non-square,
  non-tile-multiple shapes.
- cuBLAS is called via the row-major trick (swap A/B, swap M/N).

## Results

_Fill in from `results/benchmark.csv` after running on a GPU. Record GPU model and CUDA
version._

## What I learned / notes

_Write down where each optimization helped, and where it didn't._

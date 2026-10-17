# MatrixForge

MatrixForge is a CUDA matrix multiplication optimization project that implements and benchmarks FP32 SGEMM kernels against NVIDIA cuBLAS.

## Implemented Kernels

| Kernel | Optimization | Status |
|---|---|---|
| `naive` | One thread computes one output element | Implemented |
| `tiled` | Shared-memory tiling and data reuse | Implemented |
| `register_blocked` | Multiple outputs computed per thread using registers | Implemented |
| `register_blocked_tuned` | Configurable register-blocked kernel | Implemented |
| cuBLAS | NVIDIA matrix multiplication library | Performance reference |

## Project Structure

- `include/` — Kernel declarations
- `src/` — CUDA kernels, benchmark, tuning, and comparison programs
- `tests/` — Test programs
- `scripts/` — Benchmark plotting script
- `results/` — CSV measurements and generated graph
- `CMakeLists.txt` — Build configuration
- `COLAB.md` — Google Colab setup instructions

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit and `nvcc`
- CMake and a C++ compiler
- Python, pandas, and matplotlib for plotting

If you do not have a local NVIDIA GPU, use Google Colab with a GPU runtime.

## Build and Run

From the project directory:

```bash
cmake -B build
cmake --build build -j2
```

## Adaptive Kernel Selection

MatrixForge includes `register_blocked_adaptive`, a workload-aware dispatcher
that chooses between the original register-blocked kernel and the tuned
register-blocked kernel.

Current heuristic:
- Square matrices with dimension at least 384 use the tuned kernel.
- Other shapes use the original register-blocked kernel.

This is an experimentally evaluated heuristic for the tested GPU, not a
universal threshold or an automatic tuning system. Different GPUs, CUDA
versions, and workloads may need different thresholds.

## Repeated Benchmark Results

Repeated measurements were collected on an NVIDIA Tesla T4 in one Colab
session. Each matrix size was measured over five rounds, with 20 CUDA-event
timing trials per round. The table reports the median of the five round
medians. Timing variation can affect the results.

| Matrix size | Original median (ms) | Tuned median (ms) | Observed speedup |
|---:|---:|---:|---:|
| 128 | 0.0143 | 0.0266 | 0.54x |
| 256 | 0.0412 | 0.0452 | 0.91x |
| 320 | 0.0676 | 0.0635 | 1.06x |
| 384 | 0.1085 | 0.0735 | 1.48x |
| 512 | 0.2274 | 0.1317 | 1.73x |
| 640 | 0.3994 | 0.1917 | 2.08x |
| 768 | 0.5970 | 0.3164 | 1.89x |
| 1024 | 1.4930 | 0.7748 | 1.93x |
| 1280 | 3.1453 | 1.6176 | 1.94x |
| 1536 | 5.4992 | 2.8844 | 1.91x |
| 2048 | 12.9275 | 7.2178 | 1.79x |

The largest observed speedup in this repeated comparison was approximately
2.08x at 640 × 640. The tuned kernel was slower at 128 × 128 and 256 × 256,
illustrating why one kernel configuration is not optimal for every workload.

Separate held-out tests at 448, 896, and 1792 square dimensions showed the
adaptive dispatcher tracking the tuned kernel's performance closely. These
measurements are GPU- and session-specific; they do not establish that the
custom kernel universally outperforms cuBLAS.

Detailed repeated measurements are available in
`results/kernel_comparison_repeated.csv`. The standard benchmark CSV records
kernel timings, GFLOP/s, correctness error, and comparison with cuBLAS.

## Correctness and Measurement Notes

- The standard benchmark checks output correctness against a reference.
- CUDA-event timings are used for GPU kernel measurements.
- Performance depends on matrix dimensions, GPU model, compiler, and runtime.
- cuBLAS is the performance reference; results should be interpreted from
  repeated measurements rather than a single timing.

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

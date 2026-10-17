# Running on Google Colab (free GPU)

Your laptop has no NVIDIA GPU, so you write code locally (VS Code) and run it on Colab.

## 0. Get a GPU runtime
`Runtime > Change runtime type > T4 GPU`, then in a cell:

```
!nvidia-smi
!nvcc --version
```
If `nvcc` is missing, the runtime has no CUDA toolkit. Switch to a GPU runtime and retry.

## 1. Get your code into Colab
Easiest: push the project to GitHub, then

```
!git clone https://github.com/<your-username>/cuda-matmul-optimizer.git
%cd cuda-matmul-optimizer
```
(Or upload a zip with the Files panel and `!unzip`.)

## 2. Milestone 0 - does CUDA work at all?
```
!nvcc -o hello_cuda tests/hello_cuda.cu && ./hello_cuda
```
Expected: GPU name and `PASS: all 1048576 elements correct`.

## 3. Build and run the benchmark
```
!mkdir -p results
!cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build -j
!./build/matmul_bench
```
(T4 = architecture 75. If Colab gives you another GPU, check its compute
capability in the first line of the program output and adjust.)

The build prints register and shared-memory usage per kernel (`--ptxas-options=-v`).
Save those lines for your README.

## 4. Plot and save results
```
!pip -q install pandas matplotlib
!python scripts/plot_results.py
```
Colab's disk is wiped when the session ends. Download `results/benchmark.csv`
and `results/gflops.png` (Files panel), or commit them to GitHub, before closing.

## Profiling note
Nsight Compute usually cannot read hardware counters on free Colab. Do correctness
and timing first; use the `ptxas -v` output and CUDA occupancy API for register and
occupancy numbers. Only report profiler metrics you actually measured.

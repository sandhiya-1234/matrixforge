#pragma once

// Plain C++ reference implementation (no CUDA needed, compiles on your laptop).
// Row-major FP32.  A: MxK, B: KxN, C: MxN.
// Loop order i-k-j keeps the inner loop contiguous in memory, so this is a
// fairer single-threaded baseline than the textbook i-j-k order.
void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K);

#include "matmul.cuh"

// Kernel 1: naive. One thread computes one element of C.
// Every thread reads a full row of A and a full column of B from global memory.
__global__ void naive_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // x -> column: neighbouring threads
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // write/read neighbouring B and C elements

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
        }
        C[(size_t)row * N + col] = acc;
    }
}

void launch_naive(const float* dA, const float* dB, float* dC, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

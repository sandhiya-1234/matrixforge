#include "matmul.cuh"

constexpr int TILE = 16;

__global__ void tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    const int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE + threadIdx.x;
        const int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K)
                ? A[static_cast<size_t>(row) * K + a_col]
                : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N)
                ? B[static_cast<size_t>(b_row) * N + col]
                : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[static_cast<size_t>(row) * N + col] = sum;
    }
}

void launch_tiled(
    const float* dA,
    const float* dB,
    float* dC,
    int M, int N, int K)
{
    dim3 block(TILE, TILE);
    dim3 grid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    tiled_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

#include "matmul.cuh"

constexpr int RB_TILE = 16;

__global__ void register_blocked_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Each block computes up to a 32x32 output tile.
    // Each thread computes four output elements.
    __shared__ float As[2 * RB_TILE][RB_TILE];
    __shared__ float Bs[RB_TILE][2 * RB_TILE];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int row0 = blockIdx.y * (2 * RB_TILE) + ty;
    const int row1 = row0 + RB_TILE;
    const int col0 = blockIdx.x * (2 * RB_TILE) + tx;
    const int col1 = col0 + RB_TILE;

    float c00 = 0.0f, c01 = 0.0f;
    float c10 = 0.0f, c11 = 0.0f;

    const int num_tiles = (K + RB_TILE - 1) / RB_TILE;

    for (int t = 0; t < num_tiles; ++t) {
        const int kA = t * RB_TILE + tx;
        const int kB = t * RB_TILE + ty;

        As[ty][tx] =
            (row0 < M && kA < K)
                ? A[static_cast<size_t>(row0) * K + kA] : 0.0f;

        As[ty + RB_TILE][tx] =
            (row1 < M && kA < K)
                ? A[static_cast<size_t>(row1) * K + kA] : 0.0f;

        Bs[ty][tx] =
            (kB < K && col0 < N)
                ? B[static_cast<size_t>(kB) * N + col0] : 0.0f;

        Bs[ty][tx + RB_TILE] =
            (kB < K && col1 < N)
                ? B[static_cast<size_t>(kB) * N + col1] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < RB_TILE; ++k) {
            const float a0 = As[ty][k];
            const float a1 = As[ty + RB_TILE][k];
            const float b0 = Bs[k][tx];
            const float b1 = Bs[k][tx + RB_TILE];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        __syncthreads();
    }

    if (row0 < M && col0 < N)
        C[static_cast<size_t>(row0) * N + col0] = c00;

    if (row0 < M && col1 < N)
        C[static_cast<size_t>(row0) * N + col1] = c01;

    if (row1 < M && col0 < N)
        C[static_cast<size_t>(row1) * N + col0] = c10;

    if (row1 < M && col1 < N)
        C[static_cast<size_t>(row1) * N + col1] = c11;
}

void launch_register_blocked(
    const float* dA, const float* dB, float* dC,
    int M, int N, int K)
{
    dim3 block(RB_TILE, RB_TILE);
    dim3 grid(
        (N + 2 * RB_TILE - 1) / (2 * RB_TILE),
        (M + 2 * RB_TILE - 1) / (2 * RB_TILE)
    );

    register_blocked_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

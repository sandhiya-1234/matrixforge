#include "matmul.cuh"

#include <cmath>

template<int BM, int BN, int BK, int TM, int TN>
__global__ void register_blocked_v2_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int threads = blockDim.x * blockDim.y;

    const int row_base = blockIdx.y * BM + ty * TM;
    const int col_base = blockIdx.x * BN + tx * TN;

    float acc[TM][TN] = {};

    for (int tile_k = 0; tile_k < K; tile_k += BK) {
        // Cooperatively load A's BM x BK tile.
        for (int idx = tid; idx < BM * BK; idx += threads) {
            const int r = idx / BK;
            const int k = idx % BK;
            const int gr = blockIdx.y * BM + r;
            const int gk = tile_k + k;

            As[r][k] = (gr < M && gk < K)
                ? A[gr * K + gk] : 0.0f;
        }

        // Cooperatively load B's BK x BN tile.
        for (int idx = tid; idx < BK * BN; idx += threads) {
            const int k = idx / BN;
            const int c = idx % BN;
            const int gk = tile_k + k;
            const int gc = blockIdx.x * BN + c;

            Bs[k][c] = (gk < K && gc < N)
                ? B[gk * N + gc] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_vals[TM];
            float b_vals[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                a_vals[i] = As[ty * TM + i][k];

            #pragma unroll
            for (int j = 0; j < TN; ++j)
                b_vals[j] = Bs[k][tx * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_vals[i] * b_vals[j];
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int row = row_base + i;
            const int col = col_base + j;
            if (row < M && col < N)
                C[row * N + col] = acc[i][j];
        }
    }
}

template<int BM, int BN, int BK, int TM, int TN>
static void launch_config(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    register_blocked_v2_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(A, B, C, M, N, K);

    CUDA_CHECK(cudaGetLastError());
}

void launch_register_blocked_v2(
    const float* A, const float* B, float* C,
    int M, int N, int K, int config_id)
{
    switch (config_id) {
        case 0: launch_config<32, 32, 8, 2, 2>(A, B, C, M, N, K); break;
        case 1: launch_config<64, 64, 8, 4, 4>(A, B, C, M, N, K); break;
        case 2: launch_config<64, 64, 16, 4, 4>(A, B, C, M, N, K); break;
        case 3: launch_config<64, 64, 8, 4, 8>(A, B, C, M, N, K); break;
        default:
            std::fprintf(stderr, "Invalid tuning config ID: %d\n", config_id);
            std::exit(EXIT_FAILURE);
    }
}

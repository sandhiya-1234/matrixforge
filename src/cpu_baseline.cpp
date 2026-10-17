#include "cpu_baseline.h"
#include <cstring>

void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    std::memset(C, 0, sizeof(float) * (size_t)M * N);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            const float a = A[(size_t)i * K + k];
            for (int j = 0; j < N; ++j) {
                C[(size_t)i * N + j] += a * B[(size_t)k * N + j];
            }
        }
    }
}

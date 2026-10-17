#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                         cudaGetErrorString(err__), __FILE__, __LINE__);        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// All matrices are row-major FP32.  A: MxK, B: KxN, C: MxN.
// Every launcher takes DEVICE pointers and must not synchronize or time itself.

void launch_naive(const float* dA, const float* dB, float* dC, int M, int N, int K);

// TODO (next steps), same signature:
// void launch_coalesced(...);
// void launch_tiled(...);
// void launch_register_blocked(...);

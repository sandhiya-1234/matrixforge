// Benchmark driver: validates each GPU kernel against the CPU reference,
// times kernels with CUDA events (kernel-only, no host<->device copies),
// compares against cuBLAS, and appends results to a CSV.
//
// Usage: ./matmul_bench [sizes...]      e.g.  ./matmul_bench 256 512 1024
// Non-square shapes: add them to the `shapes` list below.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <cublas_v2.h>

#include "cpu_baseline.h"
#include "matmul.cuh"

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t s__ = (call);                                            \
        if (s__ != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s__,        \
                         __FILE__, __LINE__);                                   \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

static const int WARMUP = 3;
static const int TRIALS = 10;
static const int CPU_MAX_N = 1024;  // skip the CPU baseline above this size (too slow)

struct Result {
    double median_ms;
    double gflops;
};

// Time `fn` (which enqueues GPU work) with CUDA events: warm-up, then TRIALS runs, median.
static double time_gpu_ms(const std::function<void()>& fn) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < WARMUP; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> ms;
    for (int i = 0; i < TRIALS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
        ms.push_back(t);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

static double gflops(int M, int N, int K, double ms) {
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e9;
}

// Max relative error with an absolute floor, so values near zero don't blow it up.
static double max_rel_error(const std::vector<float>& ref, const std::vector<float>& got) {
    double worst = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double denom = std::max(std::fabs((double)ref[i]), 1.0);
        worst = std::max(worst, std::fabs((double)ref[i] - (double)got[i]) / denom);
    }
    return worst;
}

int main(int argc, char** argv) {
    // ---- device info (record this in your README / results) ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int rt = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));
    std::printf("GPU: %s | SMs: %d | CC %d.%d | CUDA runtime %d\n", prop.name,
                prop.multiProcessorCount, prop.major, prop.minor, rt);

    // ---- shapes to test: include sizes that are NOT multiples of the tile size ----
    std::vector<std::array<int, 3>> shapes;  // {M, N, K}
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            int n = std::atoi(argv[i]);
            shapes.push_back({n, n, n});
        }
    } else {
        for (int n : {128, 256, 512, 1024, 2048}) shapes.push_back({n, n, n});
        shapes.push_back({257, 129, 193});  // odd sizes catch boundary bugs
    }

    // ---- kernels under test: add new ones here as you write them ----
    struct Kernel {
        std::string name;
        std::function<void(const float*, const float*, float*, int, int, int)> run;
    };
    std::vector<Kernel> kernels = {
        {"naive", launch_naive},
        // {"coalesced", launch_coalesced},
        {"tiled", launch_tiled},
        // {"register_blocked", launch_register_blocked},
    };

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    FILE* csv = std::fopen("results/benchmark.csv", "w");
    if (!csv) csv = std::fopen("benchmark.csv", "w");  // fallback if results/ is missing
    std::fprintf(csv, "gpu,M,N,K,kernel,median_ms,gflops,max_rel_err,pct_of_cublas,correct\n");

    std::mt19937 rng(42);  // fixed seed -> reproducible inputs
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    for (auto& s : shapes) {
        const int M = s[0], N = s[1], K = s[2];
        std::printf("\n=== M=%d N=%d K=%d ===\n", M, N, K);

        std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N);
        for (auto& v : hA) v = dist(rng);
        for (auto& v : hB) v = dist(rng);

        // Reference result: CPU if affordable, otherwise cuBLAS (documented in output).
        std::vector<float> ref((size_t)M * N);
        bool ref_is_cpu = ((long long)M * N * K) <= (long long)CPU_MAX_N * CPU_MAX_N * CPU_MAX_N;

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, sizeof(float) * hA.size()));
        CUDA_CHECK(cudaMalloc(&dB, sizeof(float) * hB.size()));
        CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * hC.size()));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(float) * hA.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(float) * hB.size(), cudaMemcpyHostToDevice));

        // cuBLAS is column-major. Row-major C = A*B  <=>  column-major C^T = B^T * A^T,
        // so we pass B first, then A, with swapped dimensions.
        const float alpha = 1.f, beta = 0.f;
        auto run_cublas = [&]() {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N,
                                     dA, K, &beta, dC, N));
        };

        // cuBLAS timing (+ use its output as the reference for big sizes)
        double cublas_ms = time_gpu_ms(run_cublas);
        double cublas_gf = gflops(M, N, K, cublas_ms);
        CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeof(float) * hC.size(), cudaMemcpyDeviceToHost));
        if (ref_is_cpu) {
            cpu_matmul(hA.data(), hB.data(), ref.data(), M, N, K);
            std::printf("  reference: CPU baseline\n");
            std::printf("  cublas sanity vs CPU: max rel err %.2e\n", max_rel_error(ref, hC));
        } else {
            ref = hC;
            std::printf("  reference: cuBLAS (matrix too large for CPU baseline)\n");
        }
        std::fprintf(csv, "\"%s\",%d,%d,%d,cublas,%.4f,%.2f,0,100.0,1\n", prop.name, M, N, K,
                     cublas_ms, cublas_gf);
        std::printf("  %-18s %9.3f ms  %9.1f GFLOP/s  (100%% of cuBLAS)\n", "cublas", cublas_ms,
                    cublas_gf);

        // Custom kernels
        for (auto& k : kernels) {
            CUDA_CHECK(cudaMemset(dC, 0, sizeof(float) * hC.size()));
            k.run(dA, dB, dC, M, N, K);  // correctness run
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeof(float) * hC.size(), cudaMemcpyDeviceToHost));
            double err = max_rel_error(ref, hC);
            bool ok = err < 1e-3;  // FP32 accumulation-order differences; tune if needed

            double ms = time_gpu_ms([&]() { k.run(dA, dB, dC, M, N, K); });
            double gf = gflops(M, N, K, ms);
            double pct = 100.0 * gf / cublas_gf;
            std::printf("  %-18s %9.3f ms  %9.1f GFLOP/s  (%5.1f%% of cuBLAS)  err %.2e  %s\n",
                        k.name.c_str(), ms, gf, pct, err, ok ? "OK" : "WRONG");
            std::fprintf(csv, "\"%s\",%d,%d,%d,%s,%.4f,%.2f,%.3e,%.2f,%d\n", prop.name, M, N, K,
                         k.name.c_str(), ms, gf, err, pct, ok ? 1 : 0);
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    std::fclose(csv);
    CUBLAS_CHECK(cublasDestroy(handle));
    std::printf("\nResults written to results/benchmark.csv\n");
    return 0;
}

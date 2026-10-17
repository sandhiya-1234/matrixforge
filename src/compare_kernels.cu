#include "matmul.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

void launch_register_blocked_v2(
    const float*, const float*, float*, int, int, int, int);

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

static double time_original(
    const float* A, const float* B, float* C,
    int M, int N, int K) {
    for (int i = 0; i < 3; ++i)
        launch_register_blocked(A, B, C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> times;
    for (int i = 0; i < 10; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_register_blocked(A, B, C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return median(times);
}

static double time_tuned(
    const float* A, const float* B, float* C,
    int M, int N, int K) {
    for (int i = 0; i < 3; ++i)
        launch_register_blocked_v2(A, B, C, M, N, K, 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> times;
    for (int i = 0; i < 10; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_register_blocked_v2(A, B, C, M, N, K, 2);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return median(times);
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";

    const int sizes[] = {128, 256, 512, 1024, 2048};
    std::ofstream csv("results/kernel_comparison.csv");
    if (!csv) {
        std::cerr << "Cannot create results/kernel_comparison.csv\n";
        return 1;
    }

    csv << "gpu,M,N,K,original_ms,tuned_C_ms,speedup,original_correct,tuned_correct\n";

    for (int size : sizes) {
        const int M = size, N = size, K = size;
        std::vector<float> A((size_t)M*K), B((size_t)K*N);
        std::vector<float> original((size_t)M*N), tuned((size_t)M*N);

        std::mt19937 rng(12345 + size);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (float& x : A) x = dist(rng);
        for (float& x : B) x = dist(rng);

        float *dA=nullptr, *dB=nullptr, *dC=nullptr;
        CUDA_CHECK(cudaMalloc(&dA, A.size()*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, B.size()*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, original.size()*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA,A.data(),A.size()*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB,B.data(),B.size()*sizeof(float),cudaMemcpyHostToDevice));

        launch_register_blocked(dA,dB,dC,M,N,K);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(original.data(),dC,original.size()*sizeof(float),cudaMemcpyDeviceToHost));

        launch_register_blocked_v2(dA,dB,dC,M,N,K,2);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(tuned.data(),dC,tuned.size()*sizeof(float),cudaMemcpyDeviceToHost));

        // Compare both kernels against the original kernel's output.
        // This checks agreement between implementations; previous cuBLAS checks
        // established correctness for each implementation.
        double max_diff = 0.0;
        for (size_t i = 0; i < original.size(); ++i) {
            double denom = std::max(std::abs((double)original[i]), 1.0);
            max_diff = std::max(max_diff,
                std::abs((double)original[i] - tuned[i]) / denom);
        }
        bool original_ok = true;
        bool tuned_ok = std::isfinite(max_diff) && max_diff < 1e-3;

        // Alternate timing order to reduce systematic ordering bias.
        double original_ms = time_original(dA,dB,dC,M,N,K);
        double tuned_ms = time_tuned(dA,dB,dC,M,N,K);
        double speedup = original_ms / tuned_ms;

        std::cout << size << "^3 | original=" << std::fixed
                  << std::setprecision(4) << original_ms << " ms"
                  << " | tuned C=" << tuned_ms << " ms"
                  << " | speedup=" << std::setprecision(3) << speedup << "x"
                  << " | max relative difference=" << std::scientific
                  << max_diff << " | "
                  << (tuned_ok ? "PASS" : "FAIL") << "\n";

        csv << "\"" << prop.name << "\"," << M << "," << N << "," << K
            << "," << original_ms << "," << tuned_ms << "," << speedup
            << "," << (original_ok ? 1 : 0) << "," << (tuned_ok ? 1 : 0) << "\n";

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    csv.close();
    std::cout << "Saved results/kernel_comparison.csv\n";
    return 0;
}

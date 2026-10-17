
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

static double time_kernel(
    void (*launch)(const float*, const float*, float*, int, int, int),
    const float* A, const float* B, float* C,
    int M, int N, int K) {
    for (int i = 0; i < 5; ++i) launch(A, B, C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> times;
    for (int i = 0; i < 20; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch(A, B, C, M, N, K);
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

static void launch_tuned_C(
    const float* A, const float* B, float* C,
    int M, int N, int K) {
    launch_register_blocked_v2(A, B, C, M, N, K, 2);
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";

    const int sizes[] = {128, 256, 320, 384, 512, 640, 768,
                         1024, 1280, 1536, 2048};
    std::ofstream csv("results/kernel_comparison_repeated.csv");
    if (!csv) {
        std::cerr << "Cannot create results/kernel_comparison_repeated.csv\n";
        return 1;
    }

    csv << "gpu,N,original_median_ms,tuned_median_ms,speedup,"
           "original_min_ms,original_max_ms,tuned_min_ms,tuned_max_ms\n";

    for (int size : sizes) {
        int M = size, N = size, K = size;
        std::vector<float> A((size_t)M*K), B((size_t)K*N);
        std::mt19937 rng(12345 + size);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (float& x : A) x = dist(rng);
        for (float& x : B) x = dist(rng);

        float *dA=nullptr, *dB=nullptr, *dC=nullptr;
        CUDA_CHECK(cudaMalloc(&dA, A.size()*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, B.size()*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, (size_t)M*N*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA,A.data(),A.size()*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB,B.data(),B.size()*sizeof(float),cudaMemcpyHostToDevice));

        std::vector<double> orig_rounds, tuned_rounds;
        for (int round = 0; round < 5; ++round) {
            double o, t;
            if (round % 2 == 0) {
                o = time_kernel(launch_register_blocked,dA,dB,dC,M,N,K);
                t = time_kernel(launch_tuned_C,dA,dB,dC,M,N,K);
            } else {
                t = time_kernel(launch_tuned_C,dA,dB,dC,M,N,K);
                o = time_kernel(launch_register_blocked,dA,dB,dC,M,N,K);
            }
            orig_rounds.push_back(o);
            tuned_rounds.push_back(t);
        }

        double o_med = median(orig_rounds);
        double t_med = median(tuned_rounds);
        double speedup = o_med / t_med;

        auto minmax_o = std::minmax_element(orig_rounds.begin(), orig_rounds.end());
        auto minmax_t = std::minmax_element(tuned_rounds.begin(), tuned_rounds.end());

        std::cout << size << "^3 | original median=" << std::fixed
                  << std::setprecision(4) << o_med << " ms"
                  << " | tuned median=" << t_med << " ms"
                  << " | speedup=" << std::setprecision(3) << speedup << "x\n";

        csv << "\"" << prop.name << "\"," << size << "," << o_med << ","
            << t_med << "," << speedup << "," << *minmax_o.first << ","
            << *minmax_o.second << "," << *minmax_t.first << ","
            << *minmax_t.second << "\n";

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    csv.close();
    std::cout << "Saved results/kernel_comparison_repeated.csv\n";
    return 0;
}

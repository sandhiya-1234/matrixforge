#include "matmul.cuh"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

void launch_register_blocked_v2(
    const float*, const float*, float*, int, int, int, int);

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << (int)s << "\n"; \
        return 1; \
    } \
} while (0)

struct Shape { int M, N, K; };

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

double error(const std::vector<float>& a,
             const std::vector<float>& b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double denom = std::max(std::abs((double)a[i]), 1.0);
        worst = std::max(worst,
            std::abs((double)a[i] - b[i]) / denom);
    }
    return worst;
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const Shape shapes[] = {
        {128,128,128}, {256,256,256}, {512,512,512},
        {1024,1024,1024}, {2048,2048,2048},
        {257,129,193}
    };

    const char* names[] = {
        "A_32x32_BK8", "B_64x64_BK8",
        "C_64x64_BK16", "D_64x64_TM4_TN8"
    };

    std::ofstream csv("results/tuning_sweep.csv");
    if (!csv) {
        std::cerr << "Cannot create results/tuning_sweep.csv\n";
        return 1;
    }
    csv << "gpu,M,N,K,config,median_ms,gflops,max_rel_err,correct\n";

    for (const Shape s : shapes) {
        int M=s.M, N=s.N, K=s.K;
        std::vector<float> A((size_t)M*K), B((size_t)K*N);
        std::vector<float> ref((size_t)M*N), out((size_t)M*N);

        std::mt19937 rng(12345 + M + N + K);
        std::uniform_real_distribution<float> dist(-1.0f,1.0f);
        for (float& x : A) x=dist(rng);
        for (float& x : B) x=dist(rng);

        float *dA=nullptr, *dB=nullptr, *dC=nullptr;
        CUDA_CHECK(cudaMalloc(&dA,A.size()*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB,B.size()*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC,ref.size()*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA,A.data(),A.size()*sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB,B.data(),B.size()*sizeof(float),
                              cudaMemcpyHostToDevice));

        // Row-major C=A*B, using cuBLAS on transposed dimensions.
        const float alpha=1.0f, beta=0.0f;
        CUBLAS_CHECK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
            N,M,K,&alpha,dB,N,dA,K,&beta,dC,N));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(ref.data(),dC,ref.size()*sizeof(float),
                              cudaMemcpyDeviceToHost));

        for (int id=0; id<4; ++id) {
            launch_register_blocked_v2(dA,dB,dC,M,N,K,id);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(out.data(),dC,out.size()*sizeof(float),
                                  cudaMemcpyDeviceToHost));

            double err=error(ref,out);
            bool correct=std::isfinite(err) && err<1e-3;

            // Warm up, then time 10 launches with CUDA events.
            for (int i=0;i<3;++i)
                launch_register_blocked_v2(dA,dB,dC,M,N,K,id);
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaEvent_t start,stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            std::vector<double> times;

            for (int i=0;i<10;++i) {
                CUDA_CHECK(cudaEventRecord(start));
                launch_register_blocked_v2(dA,dB,dC,M,N,K,id);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                float ms=0;
                CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
                times.push_back(ms);
            }

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));

            double ms=median(times);
            double gf=2.0*M*N*K/(ms*1e6);

            std::cout << M << "x" << N << "x" << K << " | "
                      << names[id] << " | " << std::fixed
                      << std::setprecision(4) << ms << " ms | "
                      << std::setprecision(2) << gf << " GFLOP/s | "
                      << (correct ? "PASS" : "FAIL")
                      << " | error=" << std::scientific << err << "\n";

            csv << "\"" << prop.name << "\"," << M << "," << N << ","
                << K << "," << names[id] << "," << ms << "," << gf
                << "," << err << "," << (correct ? 1 : 0) << "\n";
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    csv.close();
    CUBLAS_CHECK(cublasDestroy(handle));
    std::cout << "Saved results/tuning_sweep.csv\n";
    return 0;
}

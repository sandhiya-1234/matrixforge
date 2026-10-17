// Milestone 0: prove that your environment can compile and run CUDA.
// Build:  nvcc -o hello_cuda tests/hello_cuda.cu && ./hello_cuda
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("No CUDA GPU found. In Colab: Runtime > Change runtime type > GPU.\n");
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("Using GPU: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);

    const int n = 1 << 20;
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) { a[i] = (float)i; b[i] = 2.0f * i; }

    float *da, *db, *dc;
    cudaMalloc(&da, n * sizeof(float));
    cudaMalloc(&db, n * sizeof(float));
    cudaMalloc(&dc, n * sizeof(float));
    cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    vec_add<<<(n + 255) / 256, 256>>>(da, db, dc, n);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::printf("Kernel failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost);

    int bad = 0;
    for (int i = 0; i < n; ++i)
        if (std::fabs(c[i] - 3.0f * i) > 1e-3f * std::fmax(1.0f, 3.0f * i)) ++bad;

    std::printf(bad == 0 ? "PASS: all %d elements correct\n" : "FAIL: %d wrong elements\n",
                bad == 0 ? n : bad);
    cudaFree(da); cudaFree(db); cudaFree(dc);
    return bad == 0 ? 0 : 1;
}

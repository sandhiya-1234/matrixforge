cmake_minimum_required(VERSION 3.18)
project(cuda_matmul_optimizer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# GPU architecture. "native" needs CMake >= 3.24; otherwise pass it yourself, e.g.
#   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75      (Colab T4)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
endif()

include_directories(include)

# Benchmark executable: CPU baseline + CUDA kernels + cuBLAS comparison
add_executable(matmul_bench
  src/main.cu
  src/naive.cu
  src/cpu_baseline.cpp
  src/tiled.cu
  # src/register_blocked.cu
)
target_link_libraries(matmul_bench PRIVATE cublas)
# Per-kernel register / shared-memory usage is printed at build time with this flag:
target_compile_options(matmul_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)

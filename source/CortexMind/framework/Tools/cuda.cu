//
// Created by muham on 9.05.2026.
//

#include "CortexMind/framework/Tools/cuda.cuh"
#include <CortexMind/framework/Tools/err.hpp>

namespace cortex::_fw::cuda {

    __host__ dim3 grid1d(const size_t n, const i32 block) {
        return dim3(static_cast<unsigned>((n + block - 1) / block));
    }

    __host__ dim3 grid2d(const size_t rows, const size_t cols, const i32 tile) {
        return dim3(static_cast<unsigned>((cols + tile - 1) / tile), static_cast<unsigned>((rows + tile - 1) / tile));
    }

    void memcpy(void* dst, const void* src, const size_t size, const cudaMemcpyKind kind) {
        CXM_CUDA_ASSERT(cudaMemcpy(dst, src, size, kind));
    }

    void DeviceSynchronize() {
        CXM_CUDA_ASSERT(cudaDeviceSynchronize());
    }

    void GetLastError() {
        CXM_CUDA_ASSERT(cudaGetLastError());
    }

    const char* ErrorAsString(const cudaError_t call) {
        return cudaGetErrorString(call);
    }

    void malloc(void** ptr, const size_t size) {
        CXM_CUDA_ASSERT(cudaMalloc(ptr, size));
    }

    void free(void* ptr) {
        CXM_CUDA_ASSERT(cudaFree(ptr));
    }

    void host::allocate(void** ptr, const size_t size, const unsigned int flags) {
        CXM_CUDA_ASSERT(cudaHostAlloc(ptr, size, flags));
    }

    void host::free(void* ptr) {
        CXM_CUDA_ASSERT(cudaFreeHost(ptr));
    }
} //namespace cortex::_fw::cuda
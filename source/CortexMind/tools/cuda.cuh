//
// Created by muham on 18.03.2026.
//

#ifndef CORTEXMIND_TOOLS_CUDA_CUH
#define CORTEXMIND_TOOLS_CUDA_CUH

#include <iostream>

namespace cortex {

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define CUDA_AVAILABLE_AT_COMPILE_TIME
#endif

inline bool isCudaAvailable() {
#ifdef CUDA_AVAILABLE_AT_COMPILE_TIME
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
        return false;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        if (cudaGetDeviceProperties(&deviceProp, i) != cudaSuccess) {
            return false;
        }
    }

    return true;
#else
    return false;
#endif
}

} // namespace cortex

#endif // CORTEXMIND_TOOLS_CUDA_CUH
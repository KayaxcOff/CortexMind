//
// Created by muham on 31.03.2026.
//

#ifndef CORTEXMIND_RUNTIME_CUDA_RUNTIME_CUH
#define CORTEXMIND_RUNTIME_CUDA_RUNTIME_CUH

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/framework/Tools/params.hpp>
#include <CortexMind/framework/Tools/err.hpp>
#include <cublasLt.h>
#include <cuda_runtime.h>

namespace cortex::runtime {

    inline constexpr size_t CUBLAS_WORKSPACE_SIZE = 32ull * 1024 * 1024; // 32 MB

    struct CudaContext {
        cublasLtHandle_t lt_handle  = nullptr;
        cudaStream_t     stream     = nullptr;
        void*            workspace  = nullptr;
        _fw::f32*        d_result   = nullptr;

        static CudaContext& get();

        CudaContext(const CudaContext&)            = delete;
        CudaContext& operator=(const CudaContext&) = delete;
        CudaContext(CudaContext&&)                 = delete;
        CudaContext& operator=(CudaContext&&)      = delete;

    private:
        CudaContext();
        ~CudaContext();
    };
} // namespace cortex::runtime

#endif //CORTEXMIND_RUNTIME_CUDA_RUNTIME_CUH
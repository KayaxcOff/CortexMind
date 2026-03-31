//
// Created by muham on 31.03.2026.
//

#include "CortexMind/runtime/CUDA/context.cuh"

namespace cortex::runtime {

CudaContext::CudaContext() {
    CXM_ASSERT(
        cublasLtCreate(&lt_handle) == CUBLAS_STATUS_SUCCESS,
        "cublasLtCreate failed",
        "cortex::runtime::CudaContext::CudaContext()"
    );

    CXM_ASSERT(
        cudaStreamCreate(&stream) == cudaSuccess,
        "cudaStreamCreate failed",
        "cortex::runtime::CudaContext::CudaContext()"
    );

    CXM_ASSERT(
        cudaMalloc(&workspace, CUBLAS_WORKSPACE_SIZE) == cudaSuccess,
        "workspace cudaMalloc failed",
        "cortex::runtime::CudaContext::CudaContext()"
    );

    CXM_ASSERT(
        cudaMalloc(&d_result, sizeof(float)) == cudaSuccess,
        "d_result cudaMalloc failed",
        "cortex::runtime::CudaContext::CudaContext()"
    );
}

CudaContext::~CudaContext() {
    if (d_result)   cudaFree(d_result);
    if (workspace)  cudaFree(workspace);
    if (stream)     cudaStreamDestroy(stream);
    if (lt_handle)  cublasLtDestroy(lt_handle);
}

CudaContext& CudaContext::get() {
    static CudaContext instance;
    return instance;
}

} // namespace cortex::runtime
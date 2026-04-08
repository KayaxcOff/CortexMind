//
// Created by muham on 8.04.2026.
//

#include "CortexMind/runtime/ctx.h"
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/core/Engine/CUDA/params.h>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::runtime;
using namespace cortex::_fw;

CublasContext &CublasContext::instance() {
    static CublasContext instance;
    return instance;
}

void CublasContext::init(const i32 device_id) {
    CXM_CUDA_ASSERT(cudaSetDevice(device_id), "cortex::runtime::CublasContext::init()");
    CXM_ASSERT(
        cublasCreate(&this->handle) == CUBLAS_STATUS_SUCCESS,
        "cortex::runtime::CublasContext::init()",
        "cublasCreate() failed"
    );
}

void CublasContext::destroy() const {
    CXM_ASSERT(
        cublasDestroy(this->handle) == CUBLAS_STATUS_SUCCESS,
        "cortex::runtime::CublasContext::destroy()",
        "cublasDestroy() failed"
    );
}

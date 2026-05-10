//
// Created by muham on 9.05.2026.
//

#include "CortexMind/runtime/provider.cuh"
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw::runtime;

Provider& Provider::instance() {
    static Provider provider;
    return provider;
}

Provider::Provider(const i32 device_id) {
    CXM_CUDA_ASSERT(cudaSetDevice(device_id));
    CXM_ASSERT(cublasCreate(&this->handle) == CUBLAS_STATUS_SUCCESS, "cublasCreate() failed");
}

Provider::~Provider() {
    CXM_ASSERT(cublasDestroy(this->handle) == CUBLAS_STATUS_SUCCESS, "cublasDestroy() failed");
}
//
// Created by muham on 17.08.2026.
//

#include "CortexMind/framework/Engine/CUDA/handle.cuh"
#include <CortexMind/framework/Tools/Error/errors.hpp>

using namespace cortex::_fw::nv;

handle::handle() {
    this->m_value = nullptr;

    cublasStatus_t status = cublasCreate(&this->m_value);

    CXM_ASSERT(status != CUBLAS_STATUS_SUCCESS, "Creating handle for cuBLAS has failed");
}

handle::handle(handle &&other) noexcept {
    this->m_value = other.m_value;
    other.m_value = nullptr;
}

handle::~handle() {
    if (this->m_value != nullptr) {
        cublasStatus_t status = cublasDestroy(this->m_value);
        CXM_ASSERT(status != CUBLAS_STATUS_SUCCESS, "Destroying handle for cuBLAS has failed");
    }
}

handle::operator cublasContext *() const noexcept {
    return this->m_value;
}

handle &handle::operator=(handle &&other) noexcept {
    if (this != &other) {
        if (this->m_value != nullptr) {
            cublasStatus_t status = cublasDestroy(this->m_value);
            CXM_ASSERT(status != CUBLAS_STATUS_SUCCESS, "Destroying handle for cuBLAS has failed");
        }
        this->m_value = other.m_value;
        other.m_value = nullptr;
    }
    return *this;
}
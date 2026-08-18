//
// Created by muham on 18.08.2026.
//

#include "CortexMind/framework/Engine/CUDA/stream.hpp"
#include <CortexMind/framework/Tools/Error/errors.hpp>

using namespace cortex::_fw::nv;

stream::stream() {
    this->m_value = nullptr;
    CXM_DEVICE_ASSERT(cudaStreamCreate(&this->m_value), "Creating stream has failed");
}

stream::stream(stream &&other) noexcept {
    this->m_value = other.m_value;
    other.m_value = nullptr;
}

stream::~stream() {
    if (this->m_value != nullptr) {
        CXM_DEVICE_ASSERT(cudaStreamDestroy(this->m_value), "Destroying stream has failed");
    }
}

stream::operator CUstream_st *() const noexcept {
    return this->m_value;
}

void stream::synchronize() const noexcept {
    CXM_DEVICE_ASSERT(cudaStreamSynchronize(this->m_value), "Stream synchronize has failed");
}

stream &stream::operator=(stream &&other) noexcept {
    if (this != &other) {
        if (this->m_value != nullptr) {
            CXM_DEVICE_ASSERT(cudaStreamDestroy(this->m_value), "Destroying stream has failed");
        }
        this->m_value = other.m_value;
        other.m_value = nullptr;
    }
    return *this;
}
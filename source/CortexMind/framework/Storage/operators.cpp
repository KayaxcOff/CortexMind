//
// Created by muham on 18.04.2026.
//

#include "CortexMind/framework/Storage/stor.hpp"
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Memory/transform.hpp>
#else //#if CXM_IS_CUDA_AVAILABLE
    #include <cstring>
#endif //#if CXM_IS_CUDA_AVAILABLE #else

using namespace cortex::_fw::sys;
using namespace cortex::_fw;

TensorStorage &TensorStorage::operator=(const TensorStorage &other) {
    if (this == &other) {
        return *this;
    }

    mem.deallocate(this->cpu_ptr);
    this->cpu_ptr = nullptr;
    #if CXM_IS_CUDA_AVAILABLE
        if (this->m_device == deviceType::cuda) {
            forge.deallocate(this->gpu_ptr);
            this->gpu_ptr = nullptr;
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE

    this->m_device = other.m_device;
    this->m_size   = other.m_size;

    #if CXM_IS_CUDA_AVAILABLE
        if (this->m_device == deviceType::host) {
            this->cpu_ptr = mem.allocate(this->m_size);
            transform<f32>::copy_h2h(this->cpu_ptr, other.cpu_ptr, this->m_size);
        }
        if (this->m_device == deviceType::cuda) {
            this->gpu_ptr = forge.allocate(this->m_size);
            transform<f32>::copy_d2d(this->gpu_ptr, other.gpu_ptr, this->m_size);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        std::memcpy(this->cpu_ptr, other.cpu_ptr, this->m_size * sizeof(f32));
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return *this;
}

TensorStorage &TensorStorage::operator=(TensorStorage &&other) noexcept {
    if (this == &other) {
        return *this;
    }

    mem.deallocate(this->cpu_ptr);
    this->cpu_ptr = nullptr;
    #if CXM_IS_CUDA_AVAILABLE
        if (this->m_device == deviceType::cuda) {
            forge.deallocate(this->gpu_ptr);
            this->gpu_ptr = nullptr;
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE

    this->m_device = other.m_device;
    this->m_size   = other.m_size;
    this->cpu_ptr  = other.cpu_ptr;
    this->gpu_ptr  = other.gpu_ptr;

    return *this;
}
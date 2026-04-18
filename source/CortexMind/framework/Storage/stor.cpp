//
// Created by muham on 18.04.2026.
//

#include "CortexMind/framework/Storage/stor.hpp"
#include <CortexMind/framework/Tools/err.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Memory/transform.hpp>
#else //#if CXM_IS_CUDA_AVAILABLE
    #include <cstring>
#endif //#if CXM_IS_CUDA_AVAILABLE #else

using namespace cortex::_fw::sys;
using namespace cortex::_fw;

TensorStorage::TensorStorage(const size_t size, const deviceType device) : offset(0), m_size(size), m_device(device) {
    if (this->m_device == deviceType::host) {
        this->cpu_ptr = mem.allocate(this->m_size);
    }
    if (this->m_device == deviceType::cuda) {
        #if CXM_IS_CUDA_AVAILABLE
            this->gpu_ptr = forge.allocate(this->m_size);
        #else //#if CXM_IS_CUDA_AVAILABLE
            CXM_ASSERT(false, "cortex::_fw::TensorStorage::TensorStorage()", "There is no CUDA support");
        #endif //#if CXM_IS_CUDA_AVAILABLE #else
    }
}

TensorStorage::TensorStorage(const TensorStorage &other) : offset(other.offset), m_size(other.m_size), m_device(other.m_device) {
    this->cpu_ptr = nullptr;
    this->gpu_ptr = nullptr;

    #if CXM_IS_CUDA_AVAILABLE
        if (this->m_device == deviceType::host) {
            transform<f32>::copy_h2h(this->cpu_ptr, other.cpu_ptr, this->m_size);
        }
        if (this->m_device == deviceType::cuda) {
            transform<f32>::copy_d2d(this->gpu_ptr, other.gpu_ptr, this->m_size);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        std::memcpy(this->cpu_ptr, other.cpu_ptr, this->m_size * sizeof(f32));
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

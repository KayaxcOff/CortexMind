//
// Created by muham on 18.04.2026.
//

#include "CortexMind/framework/Storage/stor.hpp"
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/framework/Tools/device_as_string.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Memory/transform.hpp>
#else //#if CXM_IS_CUDA_AVAILABLE
    #include <cstring>
#endif //#if CXM_IS_CUDA_AVAILABLE #else
#include <type_traits>

using namespace cortex::_fw::sys;
using namespace cortex::_fw;

TensorStorage::TensorStorage(const size_t size, const deviceType device) : offset(0), m_size(size), m_device(device) {
    this->cpu_ptr = nullptr;
    this->gpu_ptr = nullptr;

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

    this->m_device = other.m_device;
    this->shape = other.shape;
    this->stride = other.stride;
    this->offset = other.offset;
    this->m_size = other.m_size;

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
}

TensorStorage::TensorStorage(TensorStorage &&other) noexcept : offset(other.offset), m_size(other.m_size), m_device(other.m_device) {
    this->m_device  = other.m_device;
    this->offset    = other.offset;
    this->cpu_ptr   = other.cpu_ptr;
    this->m_size    = other.m_size;
    this->shape     = std::move(other.shape);
    this->stride    = std::move(other.stride);

    #if CXM_IS_CUDA_AVAILABLE
        this->gpu_ptr = other.gpu_ptr;
    #endif //#if CXM_IS_CUDA_AVAILABLE

    other.cpu_ptr = nullptr;
    other.gpu_ptr = nullptr;
}

TensorStorage::~TensorStorage() {
    if (this->cpu_ptr != nullptr) {
        mem.deallocate(this->cpu_ptr);
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (this->gpu_ptr != nullptr) {
            forge.deallocate(this->gpu_ptr);
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE
}

f32 *TensorStorage::data() {
    return this->m_device == deviceType::host ? this->cpu_ptr + this->offset : this->gpu_ptr + this->offset;
}

const f32 *TensorStorage::data() const {
    return this->m_device == deviceType::host ? this->cpu_ptr + this->offset : this->gpu_ptr + this->offset;
}

size_t TensorStorage::size() const noexcept {
    return this->m_size;
}

bool TensorStorage::isEmpty() const noexcept {
    return this->m_size == 0;
}

bool TensorStorage::isValid() const noexcept {
    return this->m_device == deviceType::host ? this->cpu_ptr != nullptr : this->gpu_ptr != nullptr;
}

deviceType TensorStorage::device() const noexcept {
    return this->m_device;
}

void TensorStorage::setDevice(const deviceType device) noexcept {
    if (this->m_device == device) {
        CXM_WARN(false, "cortex::_fw::TensorStorage::setDevice()",
            "Already using " + DeviceAsString(device));
        return;
    }

    #if CXM_IS_CUDA_AVAILABLE
        if (device == deviceType::cuda) {
            // host → cuda
            if (this->gpu_ptr == nullptr)
                this->gpu_ptr = forge.allocate(this->m_size);
            transform<f32>::upload(this->gpu_ptr, this->cpu_ptr, this->m_size);
        }
        if (device == deviceType::host) {
            // cuda → host
            transform<f32>::download(this->cpu_ptr, this->gpu_ptr, this->m_size);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        CXM_WARN(false, "cortex::_fw::TensorStorage::setDevice()",
            "No CUDA support, forcing CPU");
        return;
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    this->m_device = device;
}

//
// Created by muham on 10.05.2026.
//

#include "CortexMind/framework/Storage/stor.hpp"
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Memory/transform.cuh>
#else //#if CXM_IS_CUDA_AVAILABLE
    #include <cstring>
#endif //#if CXM_IS_CUDA_AVAILABLE #else
#include <CortexMind/framework/Tools/as_string.hpp>
#include <CortexMind/framework/Tools/err.hpp>
#include <iostream>

using namespace cortex::_fw::sys;
using namespace cortex::_fw;

TensorStorage::TensorStorage(const size_t _size, const DeviceType _device) : m_size(_size), m_dev(_device) {
    this->m_host_ptr = nullptr;
    this->m_cuda_ptr = nullptr;

    #if CXM_IS_CUDA_AVAILABLE
        if (this->m_dev == DeviceType::HOST) {
            this->m_host_ptr = mem.allocate(this->m_size);
        } else {
            this->m_cuda_ptr = forge.allocate(this->m_size);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        if (this->m_dev == DeviceType::HOST) {
            this->m_host_ptr = mem.allocate(this->m_size);
        } else {
            this->m_host_ptr = mem.allocate(this->m_size);
            std::cerr << "You can't use GPU so only HOST pointer allocated" << std::endl;
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

TensorStorage::TensorStorage(const TensorStorage &other) : m_size(other.m_size), m_dev(other.m_dev) {
    this->m_host_ptr = nullptr;
    this->m_cuda_ptr = nullptr;

    #if CXM_IS_CUDA_AVAILABLE
        if (this->m_dev == DeviceType::HOST) {
            this->m_host_ptr = mem.allocate(this->m_size);
            this->m_host_ptr = other.m_host_ptr;
        } else {
            this->m_cuda_ptr = forge.allocate(this->m_size);
            this->m_cuda_ptr = other.m_cuda_ptr;
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        if (this->m_dev == DeviceType::HOST) {
            this->m_host_ptr = mem.allocate(this->m_size);
            this->m_host_ptr = other.m_host_ptr;
        } else {
            this->m_host_ptr = mem.allocate(this->m_size);
            this->m_host_ptr = other.m_host_ptr;
            std::cerr << "You can't use GPU so only HOST pointer allocated" << std::endl;
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

TensorStorage::TensorStorage(TensorStorage &&other) noexcept : m_size(other.m_size), m_dev(other.m_dev) {
    this->m_host_ptr = nullptr;
    this->m_cuda_ptr = nullptr;

    #if CXM_IS_CUDA_AVAILABLE
    if (this->m_dev == DeviceType::HOST) {
        this->m_host_ptr = mem.allocate(this->m_size);
        this->m_host_ptr = other.m_host_ptr;
    } else {
        this->m_cuda_ptr = forge.allocate(this->m_size);
        this->m_cuda_ptr = other.m_cuda_ptr;
    }
    #else //#if CXM_IS_CUDA_AVAILABLE
        if (this->m_dev == DeviceType::HOST) {
            this->m_host_ptr = mem.allocate(this->m_size);
            this->m_host_ptr = other.m_host_ptr;
        } else {
            this->m_host_ptr = mem.allocate(this->m_size);
            this->m_host_ptr = other.m_host_ptr;
            std::cerr << "You can't use GPU so only HOST pointer allocated" << std::endl;
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    other.m_host_ptr = nullptr;
    other.m_cuda_ptr = nullptr;
}

TensorStorage::~TensorStorage() {
    if (this->m_host_ptr != nullptr) {
        mem.deallocate(this->m_host_ptr);
    }
    if (this->m_cuda_ptr != nullptr) {
        forge.deallocate(this->m_cuda_ptr);
    }
}

f32 *TensorStorage::data() {
    return this->m_dev == DeviceType::HOST ? this->m_host_ptr : this->m_cuda_ptr;
}

const f32 *TensorStorage::data() const {
    return this->m_dev == DeviceType::HOST ? this->m_host_ptr : this->m_cuda_ptr;
}

void TensorStorage::SetDevice(const DeviceType _device) {
    CXM_WARN(this->m_dev == _device, "You're already on " + as_string(this->m_dev) + " device");

    this->m_dev = _device;
    #if CXM_IS_CUDA_AVAILABLE
        if (this->m_dev == DeviceType::HOST) {
            if (this->m_host_ptr == nullptr) {
                this->m_host_ptr = mem.allocate(this->m_size);
            }
            transform::download(this->m_host_ptr, this->m_cuda_ptr, this->m_size);
        } else {
            if (this->m_cuda_ptr == nullptr) {
                this->m_cuda_ptr = forge.allocate(this->m_size);
            }
            transform::upload(this->m_cuda_ptr, this->m_host_ptr, this->m_size);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        std::cerr << "You can't use GPU so only HOST pointer allocated" << std::endl;
    #endif //#if CXM_IS_CUDA_AVAILABLE #else
}

size_t TensorStorage::size() const {
    return this->m_size;
}

bool TensorStorage::isEmpty() const noexcept {
    return this->m_size == 0;
}

bool TensorStorage::isValid() const noexcept {
    return this->m_dev == DeviceType::HOST ? this->m_host_ptr != nullptr : this->m_cuda_ptr != nullptr;
}

DeviceType TensorStorage::device() const noexcept {
    return this->m_dev;
}

TensorStorage TensorStorage::clone() const {
    TensorStorage output(this->m_size, this->m_dev);

    #if CXM_IS_CUDA_AVAILABLE
        if (this->m_dev == DeviceType::HOST) {
            transform::copy_h2h(output.m_host_ptr, this->m_host_ptr, this->m_size);
        } else {
            transform::copy_d2d(output.m_cuda_ptr, this->m_cuda_ptr, this->m_size);
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        std::memcpy(output.m_host_ptr, this->m_host_ptr, this->m_size * sizeof(f32));
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return output;
}
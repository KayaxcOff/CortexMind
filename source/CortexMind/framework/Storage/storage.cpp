//
// Created by muham on 3.08.2026.
//

#include "CortexMind/framework/Storage/storage.hpp"
#include <CortexMind/framework/Memory/allocators.hpp>
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Memory/transform.cuh>
#endif //#if CXM_IS_CUDA_AVAILABLE
#include <CortexMind/framework/Tools/Error/errors.hpp>

using namespace cortex::_fw;

TensorStorage::TensorStorage() {
    this->m_data = nullptr;
    this->m_bytes = 0;
    this->m_device = TensorDevice();
}

TensorStorage::TensorStorage(const std::size_t bytes, const DeviceType type) {
    this->m_bytes = bytes;
    this->m_device = TensorDevice(type);

    if (this->m_device.type() == DeviceType::HOST) {
        this->m_data = mem.allocate(this->m_bytes);
    } else if (this->m_device.type() == DeviceType::CUDA) {
        #if CXM_IS_CUDA_AVAILABLE
            this->m_data = forge.allocate(this->m_bytes);
        #else //#if CXM_IS_CUDA_AVAILABLE
            CXM_DEVICE_ERROR();
        #endif //#if CXM_IS_CUDA_AVAILABLE #else
    } else {
        CXM_DEVICE_ERROR();
    }
}

TensorStorage::TensorStorage(const std::size_t bytes, const std::byte *data, const DeviceType type) {
    this->m_bytes = bytes;
    this->m_device = TensorDevice(type);

    if (this->m_device.type() == DeviceType::HOST) {
        this->m_data = mem.allocate(this->m_bytes);
        #if CXM_IS_CUDA_AVAILABLE
            transform::copy_h2h(this->m_data, data, this->m_bytes);
        #else //#if CXM_IS_CUDA_AVAILABLE
            std::memcpy(this->m_data, data, this->m_bytes);
        #endif //#if CXM_IS_CUDA_AVAILABLE #else
    } else if (this->m_device.type() == DeviceType::CUDA) {
        #if CXM_IS_CUDA_AVAILABLE
            this->m_data = forge.allocate(this->m_bytes);
            transform::copy_d2d(this->m_data, data, this->m_bytes);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    } else {
        CXM_DEVICE_ERROR();
    }
}

TensorStorage::TensorStorage(TensorStorage &&other) noexcept {
    this->m_bytes = other.m_bytes;
    this->m_device = other.m_device;
    this->m_data = other.m_data;

    other.m_bytes = 0;
    other.m_device = TensorDevice();
    other.m_data = nullptr;
}

TensorStorage::~TensorStorage() {
    if (this->m_device.type() == DeviceType::HOST) {
        mem.deallocate(this->m_data);
    } else if (this->m_device.type() == DeviceType::CUDA) {
        #if CXM_IS_CUDA_AVAILABLE
            forge.deallocate(this->m_data);
        #endif //#if CXM_IS_CUDA_AVAILABLE
    }
}

std::byte *TensorStorage::raw() noexcept {
    return this->m_data;
}

const std::byte *TensorStorage::raw() const noexcept {
    return this->m_data;
}

std::size_t TensorStorage::bytes() const noexcept {
    return this->m_bytes;
}

bool TensorStorage::isEmpty() const noexcept {
    return this->m_bytes == 0;
}

bool TensorStorage::isValid() const noexcept {
    return this->m_data != nullptr;
}

DeviceType TensorStorage::device() const noexcept {
    return this->m_device.type();
}

TensorStorage TensorStorage::clone() const {
    return {this->m_bytes, raw(), this->m_device.type()};
}
//
// Created by muham on 3.08.2026.
//

#include "CortexMind/framework/Storage/storage.hpp"
#include <CortexMind/framework/Memory/allocators.hpp>

using namespace cortex::_fw;

TensorStorage &TensorStorage::operator=(TensorStorage &&other) noexcept {
    if (this != &other) {
        if (this->m_device.type() == DeviceType::HOST) {
            mem.deallocate(this->m_data);
        } else {
            #if CXM_IS_CUDA_AVAILABLE
                forge.deallocate(this->m_data);
            #endif //#if CXM_IS_CUDA_AVAILABLE
        }

        this->m_bytes = other.m_bytes;
        this->m_data = other.m_data;
        this->m_device = other.m_device;
        other.m_bytes = 0;
        other.m_device = TensorDevice();
        if (this->m_device.type() == DeviceType::HOST) {
            mem.deallocate(other.m_data);
        } else {
            #if CXM_IS_CUDA_AVAILABLE
                forge.deallocate(other.m_data);
            #endif //#if CXM_IS_CUDA_AVAILABLE
        }
    }
    return *this;
}
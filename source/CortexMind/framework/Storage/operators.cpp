//
// Created by muham on 10.05.2026.
//

#include "CortexMind/framework/Storage/stor.hpp"
#if !CXM_IS_CUDA_AVAILABLE
    #include <iostream>
#endif //#if !CXM_IS_CUDA_AVAILABLE

using namespace cortex::_fw::sys;
using namespace cortex::_fw;

TensorStorage &TensorStorage::operator=(const TensorStorage &other) {
    if (this == &other) {
        return *this;
    }

    this->m_size = other.m_size;
    this->m_dev = other.m_dev;

    #if CXM_IS_CUDA_AVAILABLE
        if (this->m_dev == DeviceType::kHOST) {
            this->m_host_ptr = other.m_host_ptr;
        } else {
            this->m_cuda_ptr = other.m_cuda_ptr;
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        if (this->m_dev == DeviceType::HOST) {
            this->m_host_ptr = other.m_host_ptr;
        } else {
            this->m_host_ptr = other.m_host_ptr;
            std::cerr << "You can't use GPU so only HOST pointer allocated" << std::endl;
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    return *this;
}

TensorStorage &TensorStorage::operator=(TensorStorage &&other) noexcept {
    if (this == &other) {
        return *this;
    }

    this->m_size = other.m_size;
    this->m_dev = other.m_dev;

    #if CXM_IS_CUDA_AVAILABLE
        if (this->m_dev == DeviceType::kHOST) {
            this->m_host_ptr = other.m_host_ptr;
        } else {
            this->m_cuda_ptr = other.m_cuda_ptr;
        }
    #else //#if CXM_IS_CUDA_AVAILABLE
        if (this->m_dev == DeviceType::HOST) {
            this->m_host_ptr = other.m_host_ptr;
        } else {
            this->m_host_ptr = other.m_host_ptr;
            std::cerr << "You can't use GPU so only HOST pointer allocated" << std::endl;
        }
    #endif //#if CXM_IS_CUDA_AVAILABLE #else

    other.m_host_ptr = nullptr;
    other.m_cuda_ptr = nullptr;
    other.m_size = 0;

    return *this;
}
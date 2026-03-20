//
// Created by muham on 15.03.2026.
//

#include "CortexMind/core/Engine/Storage/stor.hpp"
#include <CortexMind/core/Engine/Memory/transform.cuh>

using namespace cortex::_fw;
using namespace cortex::_fw::sys;

TensorStorage::TensorStorage(const size_t size, const dev d) : m_size(size), m_dev(d) {
    this->m_host   = static_cast<f32*>(mem.allocate(sizeof(f32) * this->m_size));
    if (d == dev::cuda)
        this->m_device = static_cast<f32*>(heap.allocate(sizeof(f32) * this->m_size));
    else
        this->m_device = nullptr;
}

TensorStorage::TensorStorage(const TensorStorage &other) : m_size(other.m_size), m_dev(other.m_dev) {
    this->m_host = static_cast<f32*>(mem.allocate(sizeof(f32) * this->m_size));
    transform<f32>::copy_h2h(this->m_host, other.m_host, sizeof(f32) * this->m_size);
    if (other.m_device) {
        this->m_device = static_cast<f32*>(heap.allocate(sizeof(f32) * this->m_size));
        transform<f32>::copy_d2d(other.m_device, this->m_device, this->m_size);
    } else {
        this->m_device = nullptr;
    }
}

TensorStorage::TensorStorage(TensorStorage &&other) noexcept {
    this->m_size    = other.m_size;
    this->m_dev     = other.m_dev;
    this->m_host    = other.m_host;
    this->m_device  = other.m_device;

    other.m_host    = nullptr;
    other.m_device  = nullptr;
}

TensorStorage::~TensorStorage() {
    mem.deallocate(this->m_host, this->m_size);
    heap.deallocate(this->m_device, this->m_size);
}

f32 *TensorStorage::data() {
    return this->m_dev == dev::host ? this->m_host : this->m_device;
}

f32 *TensorStorage::data() const {
    return this->m_dev == dev::host ? this->m_host : this->m_device;
}

size_t TensorStorage::size() const noexcept {
    return this->m_size;
}

bool TensorStorage::isEmpty() const noexcept {
    return this->m_size == 0;
}

bool TensorStorage::isValid() const {
    return this->m_host != nullptr || this->m_device != nullptr;
}

bool TensorStorage::is_cpu() const {
    return this->m_dev == dev::host;
}

bool TensorStorage::is_gpu() const {
    return this->m_dev == dev::cuda;
}

void TensorStorage::to(const dev d) {
    if (d == this->m_dev) return;
    if (d == dev::cuda && this->m_device == nullptr)
        this->m_device = static_cast<f32*>(heap.allocate(sizeof(f32) * this->m_size));
    if (d == dev::host && this->m_host == nullptr)
        this->m_host = static_cast<f32*>(mem.allocate(sizeof(f32) * this->m_size));

    if (d == dev::cuda)
        transform<f32>::upload(this->m_host, this->m_device, this->m_size);
    else
        transform<f32>::download(this->m_host, this->m_device, this->m_size);

    this->m_dev = d;
}
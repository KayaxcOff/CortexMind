//
// Created by muham on 4.02.2026.
//

#include "CortexMind/core/Engine/Storage/stor.hpp"

using namespace cortex::_fw;

TensorStorage::TensorStorage(const size_t size, std::pmr::memory_resource *resource) : m_size(size), m_alloc(resource) {
    this->m_data = this->m_alloc.allocate(size);
}

TensorStorage::TensorStorage(const TensorStorage &other) : m_size(other.m_size), m_alloc(other.m_alloc) {
    if (this->m_size > 0) {
        this->m_data = this->m_alloc.allocate(this->m_size);
        std::memcpy(this->m_data, other.m_data, this->m_size * sizeof(f32));
    }
}

TensorStorage::TensorStorage(TensorStorage &&other) noexcept :m_data(), m_size(other.m_size), m_alloc(other.m_alloc) {
    other.m_data = nullptr;
    other.m_size = 0;
}

TensorStorage::~TensorStorage() {
    if (this->m_data) {
        this->m_alloc.deallocate(this->m_data, this->m_size);
        this->m_data = nullptr;
        this->m_size = 0;
    }
}

f32* TensorStorage::data() {
    return this->m_data;
}

const f32 *TensorStorage::data() const {
    return this->m_data;
}

size_t TensorStorage::size() const {
    return this->m_size;
}

bool TensorStorage::isEmpty() const {
    return this->m_size == 0;
}

bool TensorStorage::isValid() const {
    return this->m_data != nullptr;
}
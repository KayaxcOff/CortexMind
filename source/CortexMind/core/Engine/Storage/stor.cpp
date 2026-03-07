//
// Created by muham on 22.02.2026.
//

#include "CortexMind/core/Engine/Storage/stor.hpp"
#include <CortexMind/core/Tools/err.hpp>

using namespace cortex::_fw;
using namespace cortex::_fw::sys;

TensorStorage::TensorStorage(const size_t size, const device dev) : m_data(nullptr), m_size(size), m_device(dev) {
    CXM_ASSERT(this->m_size > 0, "cortex::_fw::TensorStorage::TensorStorage()", "Size must be greater than zero");

    if (this->m_device == device::host) {
        this->allocate_cpu();
    } else if (this->m_device == device::cuda) {
        this->allocate_gpu();
    } else {
        CXM_ASSERT(false, "cortex::_fw::TensorStorage::TensorStorage()", "Invalid device type");
    }
}

TensorStorage::TensorStorage(const TensorStorage &other) : m_data(nullptr), m_size(other.m_size), m_device(other.m_device) {
    if (this->m_device == device::host) {
        this->allocate_cpu();
        std::memcpy(this->m_data, other.m_data, this->m_size * sizeof(f32));
    } else if (this->m_device == device::cuda) {
        this->allocate_gpu();
        std::vector<f32> tmp(this->m_size);
        other.m_buf->download(tmp.data(), this->m_size);
        this->m_buf->upload(tmp.data(), this->m_size);
    }
}

TensorStorage::TensorStorage(TensorStorage&& other) noexcept : m_data(other.m_data), m_buf(std::move(other.m_buf)), m_size(other.m_size), m_device(other.m_device) {
    other.m_data = nullptr;
    other.m_size    = 0;
}

TensorStorage::~TensorStorage() {
    if (this->m_device == device::host && this->m_data) {
        alloc().deallocate(this->m_data, this->m_size);
        this->m_data = nullptr;
    }
}

f32 *TensorStorage::data() {
    return this->m_data;
}

f32 *TensorStorage::data() const {
    return this->m_data;
}

buffer *TensorStorage::buf() {
    return this->m_buf.get();
}

buffer *TensorStorage::buf() const {
    return this->m_buf.get();
}

TensorStorage TensorStorage::to(const device target) const {
    if (target == this->m_device) return *this;

    TensorStorage output(this->m_size, target);
    if (this->m_device == device::host) output.m_buf->upload(this->m_data, this->m_size);
    else                        m_buf->download(output.m_data, this->m_size);
    return output;
}

size_t TensorStorage::size() const noexcept {
    return this->m_size;
}

device TensorStorage::kind() const noexcept {
    return this->m_device;
}

bool TensorStorage::is_device(const device device) const {
    return this->m_device == device;
}

bool TensorStorage::isValid() const noexcept {
    if (this->m_device == device::host) return this->m_data != nullptr;
    return this->m_buf != nullptr;
}

bool TensorStorage::isEmpty() const noexcept {
    return this->m_size == 0;
}

TensorStorage &TensorStorage::operator=(const TensorStorage &other) {
    if (this == &other) return *this;

    this->m_buf.reset();
    if (this->m_device == device::host && this->m_data) {
        alloc().deallocate(this->m_data, this->m_size);
        this->m_data = nullptr;
    }

    this->m_size   = other.m_size;
    this->m_device = other.m_device;

    if (this->m_device == device::cuda) {
        this->allocate_cpu();
        std::memcpy(this->m_data, other.m_data, this->m_size * sizeof(f32));
    } else {
        this->allocate_gpu();
        std::vector<f32> tmp(this->m_size);
        other.m_buf->download(tmp.data(), this->m_size);
        this->m_buf->upload(tmp.data(), this->m_size);
    }
    return *this;
}

TensorStorage &TensorStorage::operator=(TensorStorage &&other) noexcept {
    if (this == &other) return *this;

    if (this->m_device == device::host && this->m_data) alloc().deallocate(this->m_data, this->m_size);

    this->m_data = other.m_data;
    this->m_buf = std::move(other.m_buf);
    this->m_size = other.m_size;
    this->m_device = other.m_device;

    other.m_data = nullptr;
    other.m_size = 0;
    return *this;
}

TrackedMem &TensorStorage::mem() {
    static TrackedMem instance(256 * 1024 * 1024);
    return instance;
}

std::pmr::polymorphic_allocator<f32> TensorStorage::alloc() {
    return {&mem()};
}

void TensorStorage::allocate_cpu() {
    this->m_data = alloc().allocate(this->m_size);
}

void TensorStorage::allocate_gpu() {
    this->m_buf = std::make_unique<buffer>(this->m_size);
}

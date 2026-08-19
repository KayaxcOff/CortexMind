//
// Created by muham on 19.08.2026.
//

#include "CortexMind/framework/Tools/scratch.hpp"
#include <CortexMind/framework/Tools/Log/w.hpp>

using namespace cortex::_fw;

namespace cortex::_fw::mm {
    void* malloc(const std::size_t bytes, const std::size_t alignment) {
        return _mm_malloc(bytes, alignment);
    }
    void free(void* ptr) {
        _mm_free(ptr);
    }
} //namespace cortex::_fw::mm

Scratch::Scratch(const std::size_t size) {
    this->m_size = size;
    this->m_data = static_cast<float*>(mm::malloc(size * sizeof(float), 32));
}

Scratch::~Scratch() {
    mm::free(this->m_data);
}

float *Scratch::data() noexcept {
    return this->m_data;
}

const float *Scratch::data() const noexcept {
    return this->m_data;
}

std::size_t Scratch::size() const noexcept {
    return this->m_size;
}
//
// Created by muham on 4.08.2026.
//

#include "CortexMind/framework/Engine/AVX2/mask-runtime.hpp"
#include <CortexMind/framework/Tools/errors.hpp>

using namespace cortex::_fw::avx2;

mask::mask(const std::size_t n) {
    CXM_ASSERT(n > 8, "Runtime AVX2 mask size must be in range [0, 8]");
    this->m_size = n;
}

std::size_t mask::size() const noexcept {
    return this->m_size;
}
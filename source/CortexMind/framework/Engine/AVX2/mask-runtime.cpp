//
// Created by muham on 4.08.2026.
//

#include "CortexMind/framework/Engine/AVX2/mask-runtime.hpp"

using namespace cortex::_fw::avx2;

std::size_t mask::size() const noexcept {
    return this->m_size;
}
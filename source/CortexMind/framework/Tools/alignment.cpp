//
// Created by muham on 10.05.2026.
//

#include "CortexMind/framework/Tools/alignment.hpp"

size_t cortex::_fw::align_up(const size_t value, const size_t alignment) noexcept {
    return (value + (alignment - 1)) & ~(alignment - 1);
}
//
// Created by muham on 16.04.2026.
//

#include "CortexMind/framework/Tools/memory_utils.hpp"

size_t cortex::_fw::align_up(const size_t value, const size_t alignment) noexcept {
    return (value + (alignment - 1)) & ~(alignment - 1);
}
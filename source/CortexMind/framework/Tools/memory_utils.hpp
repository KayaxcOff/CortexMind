//
// Created by muham on 16.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_MEMORY_UTILS_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_MEMORY_UTILS_HPP

namespace cortex::_fw {
    /**
     * @brief Rounds up a value to the next multiple of the given alignment.
     *
     * This is a common operation for ensuring proper memory alignment
     * (e.g. for SIMD, CUDA, cache lines, or page boundaries).
     *
     * @param value      Value to align
     * @param alignment  Alignment boundary (must be a power of 2)
     * @return The smallest value >= `value` that is a multiple of `alignment`
     *
     * @note The implementation uses bitwise operations for efficiency.
     *       `alignment` should be a power of two for correct behavior.
     */
    [[nodiscard]]
    size_t align_up(size_t value, size_t alignment) noexcept;
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_MEMORY_UTILS_HPP
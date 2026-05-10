//
// Created by muham on 10.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_ALIGNMENT_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_ALIGNMENT_HPP

namespace cortex::_fw {
    /**
     * @brief Rounds up a value to the next multiple of the given alignment.
     *
     * This function is typically used to ensure proper memory alignment for
     * SIMD operations, CUDA allocations, or cache line alignment.
     *
     * @param value     The value to be aligned (e.g. size or offset)
     * @param alignment Desired alignment (must be power of 2)
     * @return The smallest value >= `value` that is a multiple of `alignment`
     *
     * @note `alignment` must be a power of two for correct behavior.
     * @note This is a `constexpr`-friendly, branchless implementation.
     *
     * @code
     * align_up(37, 16)  // returns 48
     * align_up(64, 64)  // returns 64
     * @endcode
     */
    [[nodiscard]]
    size_t align_up(size_t value, size_t alignment) noexcept;
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_ALIGNMENT_HPP
//
// Created by muham on 12.04.2026.
//

#ifndef CORTEXMIND_TOOLS_IS_AVX2_AVAILABLE_HPP
#define CORTEXMIND_TOOLS_IS_AVX2_AVAILABLE_HPP

namespace cortex {
    /**
     * @brief Returns whether
     * the CPU supports AVX2
     * instructions.
     */
    [[nodiscard]]
    bool is_avx2_available() noexcept;
} //namespace cortex

#endif //CORTEXMIND_TOOLS_IS_AVX2_AVAILABLE_HPP
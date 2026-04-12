//
// Created by muham on 12.04.2026.
//

#ifndef CORTEXMIND_TOOLS_IS_CUDA_AVAILABLE_HPP
#define CORTEXMIND_TOOLS_IS_CUDA_AVAILABLE_HPP

namespace cortex {
    /**
     * @brief Returns information
     * on whether CUDA is integrated
     * into CMake.
     */
    [[nodiscard]]
    bool is_cuda_available() noexcept;
} //namespace cortex

#endif //CORTEXMIND_TOOLS_IS_CUDA_AVAILABLE_HPP
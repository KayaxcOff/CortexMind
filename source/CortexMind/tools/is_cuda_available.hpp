//
// Created by muham on 15.04.2026.
//

#ifndef CORTEXMIND_TOOLS_IS_CUDA_AVAILABLE_HPP
#define CORTEXMIND_TOOLS_IS_CUDA_AVAILABLE_HPP

#include <CortexMind/tools/params.hpp>

namespace cortex {
    /**
     * @brief Returns is cuda available
     */
    [[nodiscard]]
    boolean has_cuda();
} //namespace cortex

#endif //CORTEXMIND_TOOLS_IS_CUDA_AVAILABLE_HPP
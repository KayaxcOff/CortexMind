//
// Created by muham on 20.04.2026.
//

#ifndef CORTEXMIND_TOOLS_COMPARISON_OPERATIONS_HPP
#define CORTEXMIND_TOOLS_COMPARISON_OPERATIONS_HPP

#include <CortexMind/tools/params.hpp>

namespace cortex {
    /**
     * @brief Returns the index of the first occurrence of the maximum value in the tensor.
     *
     * If the tensor is empty or contains NaN values, behavior is undefined.
     *
     * @param x Input tensor
     * @return Index of the maximum value, or -1 if not found (should not happen for valid tensors)
     */
    [[nodiscard]]
    int32 argmax(const tensor& x);
    /**
     * @brief Returns the index of the first occurrence of the minimum value in the tensor.
     *
     * If the tensor is empty or contains NaN values, behavior is undefined.
     *
     * @param x Input tensor
     * @return Index of the minimum value, or -1 if not found (should not happen for valid tensors)
     */
    [[nodiscard]]
    int32 argmin(const tensor& x);
} //namespace cortex

#endif //CORTEXMIND_TOOLS_COMPARISON_OPERATIONS_HPP
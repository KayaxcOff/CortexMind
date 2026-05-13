//
// Created by muham on 13.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HPP

#include <CortexMind/framework/Tools/types.hpp>
#include <vector>

namespace cortex::_fw {
    /**
     * @brief Computes the stride vector for a given tensor shape (row-major order).
     *
     * Stride represents how many elements to skip in memory to move to the next
     * index along each dimension.
     *
     * @param shape Tensor shape (e.g. {2, 3, 4})
     * @return Stride vector (e.g. {12, 4, 1} for shape {2,3,4})
     *
     * @note Assumes row-major (C-style) memory layout.
     * @note Returns empty vector if shape is empty.
     */
    [[nodiscard]]
    std::vector<i64> compute_stride(const std::vector<i64>& shape);
    /**
     * @brief Computes the total number of elements in a tensor.
     *
     * @param shape Tensor shape (dimensions)
     * @return Total number of elements (`product of all dimensions`)
     *
     * @note Returns 1 for empty shape (scalar case).
     * @note Uses `std::accumulate` with `std::multiplies` internally.
     */
    [[nodiscard]]
    size_t compute_size(const std::vector<i64>& shape);
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HPP
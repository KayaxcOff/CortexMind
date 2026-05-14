//
// Created by muham on 13.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HPP

#include <CortexMind/framework/Tools/broadcast_kind.hpp>
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
    /**
     * @brief Checks whether two shapes are broadcastable according to broadcasting rules.
     *
     * Follows NumPy/PyTorch style broadcasting rules.
     *
     * @param shape_x First shape
     * @param shape_y Second shape
     * @return `true` if shapes are broadcastable
     */
    [[nodiscard]]
    bool is_broadcastable(const std::vector<i64>& shape_x, const std::vector<i64>& shape_y);
    /**
     * @brief Classifies the type of broadcast needed between two shapes.
     *
     * @param shape_x First shape
     * @param shape_y Second shape
     * @return BroadcastKind (None, Row, Col, General)
     */
    [[nodiscard]]
    BroadcastKind classify_broadcast(const std::vector<i64>& shape_x, const std::vector<i64>& shape_y);
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HPP
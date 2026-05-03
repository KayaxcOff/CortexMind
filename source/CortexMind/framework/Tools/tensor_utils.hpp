//
// Created by muham on 18.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_UTILS_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_UTILS_HPP

#include <CortexMind/core/Tools/broadcast.hpp>
#include <CortexMind/framework/Tools/params.hpp>
#include <vector>

namespace cortex::_fw {
    /**
     * @brief Computes the default contiguous stride for a given shape.
     *
     * Stride is calculated from the last dimension to the first (row-major / C-style ordering).
     *
     * @param shape Tensor shape (dimensions)
     * @return Vector containing the stride for each dimension
     */
    [[nodiscard]]
    std::vector<i64> compute_stride(const std::vector<i64>& shape);
    /**
     * @brief Computes the total number of elements in a tensor (product of all dimensions).
     * @param shape Tensor shape
     * @return Total number of elements (numel)
     */
    [[nodiscard]]
    size_t compute_numel(const std::vector<i64>& shape);
    /**
     * @brief Computes the linear offset for a given index using the provided stride.
     *
     * This is used for converting multidimensional indices to a flat memory offset.
     *
     * @param index  Multi-dimensional index
     * @param stride Stride vector for each dimension
     * @return Linear offset in the underlying storage
     */
    [[nodiscard]]
    i64 compute_offset(const std::vector<i64>& index, const std::vector<i64>& stride);
    /**
     * @brief Checks whether the tensor is contiguous in memory.
     *
     * A tensor is contiguous if its stride matches the default row-major stride
     * (i.e., no gaps or padding between elements).
     *
     * @param shape  Tensor shape
     * @param stride Tensor stride
     * @return `true` if the tensor is contiguous, `false` otherwise
     */
    [[nodiscard]]
    bool is_contiguous(const std::vector<i64>& shape, const std::vector<i64>& stride);

    /**
     * @brief Checks if two shapes are broadcastable and computes BroadcastInfo.
     *
     * Follows NumPy broadcasting rules:
     * - Shapes are aligned from the right
     * - A dimension of 1 can be broadcast to any size
     * - Missing dimensions are treated as 1
     *
     * @param shape_x Shape of first tensor
     * @param shape_y Shape of second tensor
     * @return BroadcastInfo containing output shape and strides
     */
    [[nodiscard]]
    BroadcastInfo compute_broadcast(const std::vector<i64>& shape_x, const std::vector<i64>& shape_y);

    /**
     * @brief Checks if two shapes are equal (no broadcast needed).
     */
    [[nodiscard]]
    bool shapes_equal(const std::vector<i64>& shape_x, const std::vector<i64>& shape_y);

    /**
     * @brief Checks if two shapes are broadcastable without computing full info.
     */
    [[nodiscard]]
    bool is_broadcastable(const std::vector<i64>& shape_x, const std::vector<i64>& shape_y);
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_UTILS_HPP
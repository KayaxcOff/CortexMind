//
// Created by muham on 13.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HPP

#include <CortexMind/framework/Tools/broadcast_info.hpp>
#include <CortexMind/framework/Tools/broadcast_kind.hpp>
#include <CortexMind/framework/Tools/types.hpp>
#include <CortexMind/runtime/macros.hpp>
#include <array>

namespace cortex::_fw {
    /**
     * @brief Computes the stride vector for a given tensor shape (row-major order).
     *
     * Stride represents how many elements to skip in memory to move to the next
     * index along each dimension.
     *
     * @param shape Tensor shape (e.g. {2, 3, 4})
     * @param ndim  Dimension of tensor
     * @return Stride vector (e.g. {12, 4, 1} for shape {2,3,4})
     *
     * @note Assumes row-major (C-style) memory layout.
     * @note Returns empty vector if shape is empty.
     */
    [[nodiscard]]
    std::array<i64, CXM_MAX_DIMS> compute_stride(const std::array<i64, CXM_MAX_DIMS>& shape, size_t ndim);
    /**
     * @brief Computes the total number of elements in a tensor.
     *
     * @param shape Tensor shape (dimensions)
     * @param ndim  Dimension of tensor
     * @return Total number of elements (`product of all dimensions`)
     */
    [[nodiscard]]
    size_t compute_size(const std::array<i64, CXM_MAX_DIMS>& shape, size_t ndim);
    [[nodiscard]]
    i64 compute_idx(const std::array<i64, CXM_MAX_DIMS>& strides, const std::array<i64, CXM_MAX_DIMS>& indices, i32 ndim, i64 offset = 0);
    [[nodiscard]] bool is_contiguous(const std::array<i64, CXM_MAX_DIMS>& strides, const std::array<i64, CXM_MAX_DIMS>& shape, i32 ndim);
    /*
    [[nodiscard]]
    bool is_broadcastable(const std::vector<i64>& shape_x, const std::vector<i64>& shape_y);
    [[nodiscard]]
    std::vector<i64> broadcast_shape(const std::vector<i64>& shape_x, const std::vector<i64>& shape_y);
    [[nodiscard]]
    BroadcastKind classify_broadcast(const std::vector<i64>& shape_x, const std::vector<i64>& shape_y);
    [[nodiscard]]
    BroadcastInfo make_broadcast_info(const std::vector<i64> &shape_a, const std::vector<i64> &stride_a, const std::vector<i64> &shape_b, const std::vector<i64> &stride_b, const std::vector<i64> &shape_z, const std::vector<i64> &stride_z);
    [[nodiscard]]
    bool is_contiguous(const std::vector<i64>& strides, const std::vector<i64>& shape);
    [[nodiscard]]
    std::vector<i64> grad_reduce_dims(const std::vector<i64>& input_shape, const std::vector<i64>& grad_shape);
    */
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HPP
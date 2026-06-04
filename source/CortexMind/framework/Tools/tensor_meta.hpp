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
     * @return Total number of elements (`product of all dimensions`)
     */
    [[nodiscard]]
    size_t compute_size(const std::array<i64, CXM_MAX_DIMS>& shape, size_t ndim);
    /**
     * @brief Checks whether two shapes can be broadcasted together.
     *
     * Follows standard broadcasting rules (NumPy/PyTorch style).
     *
     * @param shape_x First tensor shape
     * @param shape_y Second tensor shape
     * @return `true` if the shapes are broadcastable
     */
    [[nodiscard]]
    bool is_broadcastable(const std::vector<i64>& shape_x, const std::vector<i64>& shape_y);
    /**
     * @brief Computes the resulting shape after broadcasting two shapes.
     *
     * @param shape_x First tensor shape
     * @param shape_y Second tensor shape
     * @return The broadcasted common shape
     */
    [[nodiscard]]
    std::vector<i64> broadcast_shape(const std::vector<i64>& shape_x, const std::vector<i64>& shape_y);
    /**
     * @brief Classifies the type of broadcasting needed between two shapes.
     *
     * @param shape_x First tensor shape
     * @param shape_y Second tensor shape
     * @return Broadcast kind (`kNone`, `kRow`, `kCol`, `kGeneral`)
     */
    [[nodiscard]]
    BroadcastKind classify_broadcast(const std::vector<i64>& shape_x, const std::vector<i64>& shape_y);
    /**
     * @brief Creates a `BroadcastInfo` structure for efficient kernel execution.
     *
     * Precomputes shape and stride information to avoid repeated calculations
     * inside hot loops during broadcasting operations.
     *
     * @param shape_a  Shape of first input
     * @param stride_a Strides of first input
     * @param shape_b  Shape of second input
     * @param stride_b Strides of second input
     * @param shape_z  Broadcasted output shape
     * @param stride_z Strides of output
     * @return Fully filled `BroadcastInfo` struct
     */
    [[nodiscard]]
    BroadcastInfo make_broadcast_info(const std::vector<i64> &shape_a, const std::vector<i64> &stride_a, const std::vector<i64> &shape_b, const std::vector<i64> &stride_b, const std::vector<i64> &shape_z, const std::vector<i64> &stride_z);
    /**
     * @brief Computes the linear (flat) memory index from multidimensional indices.
     *
     * Converts a multidimensional index into a single linear offset in memory
     * using the provided strides (row-major layout).
     *
     * @param strides Pre-computed stride vector for the tensor
     * @param indices Multi-dimensional index (same size as tensor rank)
     * @param offset  Base offset (usually 0, used for views/slices)
     * @return Linear index in the underlying storage
     *
     * @note Assumes row-major memory layout.
     * @note No bounds checking is performed for performance reasons.
     */
    [[nodiscard]]
    i64 compute_linear_index(const std::vector<i64>& strides, const std::vector<i64>& indices, i64 offset);
    /**
     * @brief Checks whether a tensor is contiguous in memory.
     *
     * A tensor is contiguous if its strides match the default row-major strides
     * computed from its shape (no gaps or non-standard layout).
     *
     * @param strides Current stride vector of the tensor
     * @param shape   Shape of the tensor
     * @return `true` if the tensor is contiguous, `false` otherwise
     *
     * @note Used to determine whether operations like `reshape`, `view`, or
     *       direct pointer arithmetic can be safely performed.
     */
    [[nodiscard]]
    bool is_contiguous(const std::vector<i64>& strides, const std::vector<i64>& shape);
    [[nodiscard]]
    std::vector<i64> grad_reduce_dims(const std::vector<i64>& input_shape, const std::vector<i64>& grad_shape);
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HPP
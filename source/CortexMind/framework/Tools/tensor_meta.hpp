//
// Created by muham on 13.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HPP

#include <CortexMind/framework/Shape/shape.hpp>
#include <CortexMind/framework/Tools/broadcast_info.hpp>
#include <CortexMind/framework/Tools/broadcast_kind.hpp>
#include <CortexMind/framework/Tools/types.hpp>
#include <vector>

namespace cortex::_fw {
    [[nodiscard]]
    std::vector<i64> compute_stride(const std::vector<i64>& shape);
    [[nodiscard]]
    size_t compute_size(const std::vector<i64>& shape);
    [[nodiscard]]
    i64 compute_idx(const std::vector<i64>& stride, const std::vector<i64>& indices, i64 offset);
    [[nodiscard]]
    bool is_contiguous(const std::vector<i64>& shape, const std::vector<i64>& stride);
    [[nodiscard]]
    bool is_broadcastable(const TensorShape& shape_x, const TensorShape& shape_y);
    [[nodiscard]]
    TensorShape broadcast_shape(const TensorShape& shape_x, const TensorShape& shape_y);
    [[nodiscard]]
    BroadcastKind classify_broadcast(const TensorShape& shape_x, const TensorShape& shape_y);
    [[nodiscard]]
    BroadcastInfo make_broadcast_info(const TensorShape& shape_a, const TensorShape& shape_b, const TensorShape& shape_z);
    [[nodiscard]]
    std::vector<i64> grad_reduce_dims(const TensorShape& input_shape, const TensorShape& grad_shape);
    [[nodiscard]]
    std::vector<i64> compute_reduced_shape(const TensorShape& current_shape, const std::vector<i64>& dims, bool keep_dim);
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HPP
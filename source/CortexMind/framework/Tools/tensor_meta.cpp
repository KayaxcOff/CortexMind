//
// Created by muham on 13.05.2026.
//

#include "CortexMind/framework/Tools/tensor_meta.hpp"
#include <CortexMind/framework/Tools/err.hpp>
#include <numeric>
#include <utility>
#include <xutility>

#include <iostream>

using namespace cortex::_fw;

std::vector<i64> cortex::_fw::compute_stride(const std::vector<i64> &shape) {
    CXM_ASSERT(shape.empty(), "Shape is empty");

    const i32 ndim = static_cast<i32>(shape.size());
    std::vector<i64> output(ndim);

    output[ndim - 1] = 1;

    for (i32 i = ndim - 2; i >= 0; --i) {
        output[i] = output[i + 1] * shape[i + 1];
    }

    return output;
}

size_t cortex::_fw::compute_size(const std::vector<i64> &shape) {
    CXM_ASSERT(shape.empty(), "Shape is empty");

    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies());
}

i64 cortex::_fw::compute_idx(const std::vector<i64>& stride, const std::vector<i64>& indices, const i64 offset) {
    CXM_ASSERT(stride.size() != indices.size(), "Strides and indices size mismatch");

    i64 output = offset;

    for (size_t i = 0; i < indices.size(); ++i) {
        output += indices[i] * stride[i];
    }

    return output;
}

bool cortex::_fw::is_contiguous(const std::vector<i64> &shape, const std::vector<i64> &stride) {
    CXM_ASSERT(shape.size() != stride.size(), "Shape and stride sizes mismatch");
    CXM_ASSERT(shape.empty(), "Shape is empty");

    i64 expected_stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        if (shape[i] == 1) {
            continue;
        }

        if (stride[i] != expected_stride) {
            return false;
        }

        expected_stride *= shape[i];
    }

    return true;
}

bool cortex::_fw::is_broadcastable(const TensorShape &shape_x, const TensorShape &shape_y) {
    const size_t rank_x = shape_x.shape.size();
    const size_t rank_y = shape_y.shape.size();
    const size_t max_rank = std::max(rank_x, rank_y);

    for (size_t i = 0; i < max_rank; ++i) {
        const i64 dim_x = (i < rank_x) ? shape_x.shape[rank_x - 1 - i] : 1;
        const i64 dim_y = (i < rank_y) ? shape_y.shape[rank_y - 1 - i] : 1;

        if (dim_x != dim_y && dim_x != 1 && dim_y != 1) {
            return false;
        }
    }
    return true;
}

TensorShape cortex::_fw::broadcast_shape(const TensorShape &shape_x, const TensorShape &shape_y) {
    const size_t rank_x = shape_x.shape.size();
    const size_t rank_y = shape_y.shape.size();
    const size_t max_rank = std::max(rank_x, rank_y);

    TensorShape output;
    output.shape.resize(max_rank);

    for (size_t i = 0; i < max_rank; ++i) {
        const i64 da = (i < rank_x) ? shape_x.shape[rank_x - 1 - i] : 1;
        const i64 db = (i < rank_y) ? shape_y.shape[rank_y - 1 - i] : 1;

        output.shape[max_rank - 1 - i] = std::max(da, db);
    }

    return output;
}

BroadcastKind cortex::_fw::classify_broadcast(const TensorShape &shape_x, const TensorShape &shape_y) {
    if (!is_broadcastable(shape_x, shape_y)) {
        return BroadcastKind::kNone;
    }

    const size_t rank_x = shape_x.shape.size();
    const size_t rank_y = shape_y.shape.size();

    if (shape_x.shape == shape_y.shape) {
        return BroadcastKind::kNone;
    }

    if (rank_y == 1 && rank_x >= 1 && shape_y.shape[0] == shape_x.shape[rank_x - 1]) {
        return BroadcastKind::kRow;
    }
    if (rank_x == 1 && rank_y >= 1 && shape_x.shape[0] == shape_y.shape[rank_y - 1]) {
        return BroadcastKind::kRow;
    }

    if (rank_x == 2 && rank_y == 2 && shape_x.shape[0] == shape_y.shape[0] && shape_y.shape[1] == 1) {
        return BroadcastKind::kCol;
    }
    if (rank_x == 2 && rank_y == 2 && shape_y.shape[0] == shape_x.shape[0] && shape_x.shape[1] == 1) {
        return BroadcastKind::kCol;
    }

    return BroadcastKind::kGeneral;
}

BroadcastInfo cortex::_fw::make_broadcast_info(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_z) {
    BroadcastInfo output{};

    const size_t ndim = shape_z.shape.size();
    CXM_ASSERT(ndim > CXM_MAX_DIMS, "Dimension is higher tan limit");

    output.ndim = static_cast<i32>(ndim);

    const i64 off_a = static_cast<i64>(ndim) - static_cast<i64>(shape_a.shape.size());
    const i64 off_b = static_cast<i64>(ndim) - static_cast<i64>(shape_b.shape.size());

    for (size_t d = 0; d < ndim; ++d) {
        output.shape[d] = shape_z.shape[d];

        const i64 da = static_cast<i64>(d) - off_a;
        output.stride_x[d] = (da >= 0 && shape_a.shape[da] != 1) ? shape_a.stride[da] : 0;

        const i64 db = static_cast<i64>(d) - off_b;
        output.stride_y[d] = (db >= 0 && shape_b.shape[db] != 1) ? shape_b.stride[db] : 0;

        output.stride_z[d] = shape_z.stride[d];
    }

    return output;
}

std::vector<i64> cortex::_fw::grad_reduce_dims(const TensorShape &input_shape, const TensorShape &grad_shape) {
    std::vector<i64> output;

    const size_t rank_in = input_shape.shape.size();
    const size_t rank_grad = grad_shape.shape.size();

    const i64 offset = static_cast<i64>(rank_grad) - static_cast<i64>(rank_in);

    for (size_t d = 0; d < rank_grad; ++d) {
        if (static_cast<i64>(d) < offset) {
            output.push_back(static_cast<i64>(d));
        } else {
            const size_t id = d - static_cast<i64>(offset);
            if (input_shape.shape[id] == 1 && grad_shape.shape[d] > 1) {
                output.push_back(static_cast<i64>(d));
            }
        }
    }
    return output;
}

std::vector<i64> cortex::_fw::compute_reduced_shape(const TensorShape &current_shape, const std::vector<i64> &dims, const bool keep_dim) {
    const size_t ndim = current_shape.shape.size();
    std::vector reduce_mask(ndim, false);

    for(const auto item : dims) {
        reduce_mask[(item < 0) ? item + ndim : item] = true;
    }

    std::vector<i64> output;
    for (size_t i = 0; i < ndim; ++i) {
        if (reduce_mask[i]) {
            if (keep_dim) {
                output.push_back(1);
            }
        } else {
            output.push_back(current_shape.shape[i]);
        }
    }
    if (output.empty()) {
        output.push_back(1);
    }
    return output;
}

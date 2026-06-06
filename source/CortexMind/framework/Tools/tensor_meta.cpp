//
// Created by muham on 13.05.2026.
//

#include "CortexMind/framework/Tools/tensor_meta.hpp"
#include <CortexMind/framework/Tools/err.hpp>
#include <numeric>
#include <utility>
#include <xutility>

using namespace cortex::_fw;

std::array<i64, 8> cortex::_fw::compute_stride(const std::array<i64, 8> &shape, const size_t ndim) {
    std::array<i64, CXM_MAX_DIMS> output{};

    if (ndim <= 0) {
        return output;
    }

    output[ndim - 1] = 1;

    for (i32 i = static_cast<i32>(ndim) - 2; i >= 0; --i) {
        output[i] = output[i + 1] * shape[i + 1];
    }

    return output;
}

size_t cortex::_fw::compute_size(const std::array<i64, 8> &shape, const size_t ndim) {
    if (ndim <= 0) {
        return 0;
    }

    return std::accumulate(shape.begin(), shape.begin() + static_cast<long long>(ndim), size_t{1}, std::multiplies());
}

i64 cortex::_fw::compute_idx(const std::array<i64, 8> &strides, const std::array<i64, 8> &indices, const i32 ndim, const i64 offset) {
    CXM_ASSERT(ndim >= CXM_MAX_DIMS, "Dimension count is too large");

    i64 output = offset;

    for (i32 i = 0; i < ndim; ++i) {
        output += indices[i] * strides[i];
    }

    return output;
}

bool cortex::_fw::is_contiguous(const std::array<i64, 8> &strides, const std::array<i64, 8> &shape, const i32 ndim) {
    if (ndim <= 1) {
        return true;
    }

    i64 expected_stride = 1;
    for (i32 i = ndim - 1; i >= 0; --i) {
        if (shape[i] == 1) {
            continue;
        }

        if (strides[i] != expected_stride) {
            return false;
        }

        expected_stride *= shape[i];
    }

    return true;
}

bool cortex::_fw::is_broadcastable(const TensorShape &shape_x, const TensorShape &shape_y) {
    const i32 max_rank = std::max(shape_x.ndim, shape_y.ndim);

    for (i32 i = 0; i < max_rank; ++i) {
        const i64 dim_x = (i < shape_x.ndim) ? shape_x.shape[shape_x.ndim - 1 - i] : 1;
        const i64 dim_y = (i < shape_y.ndim) ? shape_y.shape[shape_y.ndim - 1 - i] : 1;

        if (dim_x != dim_y && dim_x != 1 && dim_y != 1) {
            return false;
        }
    }
    return true;
}

TensorShape cortex::_fw::broadcast_shape(const TensorShape &shape_x, const TensorShape &shape_y) {
    TensorShape output{};
    output.ndim = std::max(shape_x.ndim, shape_y.ndim);
    output.offset = 0;

    for (i32 i = 0; i < output.ndim; ++i) {
        const i64 da = (i < shape_x.ndim) ? shape_x.shape[shape_x.ndim - 1 - i] : 1;
        const i64 db = (i < shape_y.ndim) ? shape_y.shape[shape_y.ndim - 1 - i] : 1;

        output.shape[output.ndim - 1 - i] = std::max(da, db);
    }

    return output;
}

BroadcastKind cortex::_fw::classify_broadcast(const TensorShape &shape_x, const TensorShape &shape_y) {
    if (!is_broadcastable(shape_x, shape_y)) {
        return BroadcastKind::kNone;
    }

    if (shape_x.ndim == shape_y.ndim) {
        bool identical = true;
        for (i32 i = 0; i < shape_x.ndim; ++i) {
            if (shape_x.shape[i] != shape_y.shape[i]) {
                identical = false;
                break;
            }
        }
        if (identical) {
            return BroadcastKind::kNone;
        }
    }

    if (shape_x.ndim == 1 && shape_x.shape[0] == 1) {
        return BroadcastKind::kGeneral;
    }
    if (shape_y.ndim == 1 && shape_y.shape[0] == 1) {
        return BroadcastKind::kGeneral;
    }

    if (shape_y.ndim == 1 && shape_y.shape[0] == shape_x.shape[shape_x.ndim - 1]) {
        return BroadcastKind::kRow;
    }

    if (shape_x.ndim == 2 && shape_y.ndim == 2 && shape_y.shape[1] == 1 && shape_x.shape[0] == shape_y.shape[0]) {
        return BroadcastKind::kCol;
    }

    if (shape_x.ndim == 1 && shape_x.shape[0] == shape_y.shape[0] && shape_y.shape[shape_y.ndim - 1] != shape_x.shape[0]) {
        return BroadcastKind::kCol;
    }

    return BroadcastKind::kGeneral;
}

BroadcastInfo cortex::_fw::make_broadcast_info(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_z) {
    BroadcastInfo output{};
    output.ndim = shape_z.ndim;

    const i32 off_a = output.ndim - shape_a.ndim;
    const i32 off_b = output.ndim - shape_b.ndim;

    for (i32 d = 0; d < output.ndim; ++d) {
        output.shape[d] = shape_z.shape[d];

        const i32 da = d - off_a;
        output.stride_x[d] = (da >= 0 && shape_a.shape[da] != 1) ? shape_a.stride[da] : 0;

        const i32 db = d - off_b;
        output.stride_y[d] = (db >= 0 && shape_b.shape[db] != 1) ? shape_b.stride[db] : 0;

        output.stride_z[d] = shape_z.stride[d];
    }
    return output;
}

std::vector<i64> cortex::_fw::grad_reduce_dims(const TensorShape &input_shape, const TensorShape &grad_shape) {
    std::vector<i64> output;
    const i32 ndim = grad_shape.ndim;
    const i32 offset = ndim - input_shape.ndim;

    for (i32 d = 0; d < ndim; ++d) {
        if (d < offset) {
            output.push_back(d);
        } else {
            const i32 id = d - offset;
            if (input_shape.shape[id] == 1 && grad_shape.shape[d] > 1) {
                output.push_back(d);
            }
        }
    }
    return output;
}

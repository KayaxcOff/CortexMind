//
// Created by muham on 13.05.2026.
//

#include "CortexMind/framework/Tools/tensor_meta.hpp"
#include <numeric>
#include <utility>
#include <xutility>

using namespace cortex::_fw;

std::vector<i64> cortex::_fw::compute_stride(const std::vector<i64> &shape) {
    std::vector<i64> output(shape.size());

    if (shape.empty()) {
        return output;
    }

    output.back() = 1;

    for (i32 i = static_cast<i32>(shape.size()) - 2; i >= 0; --i) {
        output[i] = output[i + 1] * shape[i + 1];
    }

    return output;
}

size_t cortex::_fw::compute_size(const std::vector<i64> &shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies());
}

bool cortex::_fw::is_broadcastable(const std::vector<i64>& shape_x, const std::vector<i64>& shape_y) {
    const size_t rank_x = shape_x.size();
    const size_t rank_y = shape_y.size();
    const size_t max_rank = std::max(rank_x, rank_y);

    for (size_t i = 0; i < max_rank; ++i) {
        const i64 dim_x = (i < rank_x) ? shape_x[rank_x - 1 - i] : 1;
        const i64 dim_y = (i < rank_y) ? shape_y[rank_y - 1 - i] : 1;

        if (dim_x != dim_y && dim_x != 1 && dim_y != 1) {
            return false;
        }
    }
    return true;
}

std::vector<i64> cortex::_fw::broadcast_shape(const std::vector<i64> &shape_x, const std::vector<i64> &shape_y) {
    const size_t ndim = std::max(shape_x.size(), shape_y.size());
    std::vector<i64> out(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        const i64 da = i < shape_x.size() ? shape_x[shape_x.size() - 1 - i] : 1;
        const i64 db = i < shape_y.size() ? shape_y[shape_y.size() - 1 - i] : 1;
        out[ndim - 1 - i] = std::max(da, db);
    }
    return out;
}

BroadcastKind cortex::_fw::classify_broadcast(const std::vector<i64>& shape_x, const std::vector<i64>& shape_y) {
    if (!is_broadcastable(shape_x, shape_y)) {
        return BroadcastKind::kNone;
    }

    if (shape_x == shape_y) {
        return BroadcastKind::kNone;
    }

    const size_t rank_x = shape_x.size();
    const size_t rank_y = shape_y.size();

    if (rank_y == 1 && shape_y[0] == shape_x.back()) {
        return BroadcastKind::kRow;
    }

    if (rank_x == 1 && shape_x[0] == shape_y.back()) {
        return BroadcastKind::kCol;
    }

    return BroadcastKind::kGeneral;
}

BroadcastInfo cortex::_fw::make_broadcast_info(const std::vector<i64>& shape_a, const std::vector<i64>& stride_a, const std::vector<i64>& shape_b, const std::vector<i64>& stride_b, const std::vector<i64>& shape_z, const std::vector<i64>& stride_z){
    BroadcastInfo info{};
    info.ndim = static_cast<i32>(shape_z.size());

    const i32 off_a = info.ndim - static_cast<i32>(shape_a.size());
    const i32 off_b = info.ndim - static_cast<i32>(shape_b.size());

    for (i32 d = 0; d < info.ndim; ++d) {
        info.shape[d] = static_cast<size_t>(shape_z[d]);

        const i32 da = d - off_a;
        info.stride_x[d] = (da >= 0 && shape_a[da] != 1)
            ? static_cast<size_t>(stride_a[da]) : 0;

        const i32 db = d - off_b;
        info.stride_y[d] = (db >= 0 && shape_b[db] != 1)
            ? static_cast<size_t>(stride_b[db]) : 0;

        info.stride_z[d] = static_cast<size_t>(stride_z[d]);
    }
    return info;
}

i64 cortex::_fw::compute_linear_index(const std::vector<i64> &strides, const std::vector<i64> &indices, i64 offset) {
    i64 output = offset;
    for (size_t d = 0; d < strides.size(); ++d) {
        output += indices[d] * strides[d];
    }
    return output;
}

// tensor_meta.cpp
bool cortex::_fw::is_contiguous(const std::vector<i64>& strides, const std::vector<i64>& shape) {
    if (strides.size() != shape.size()) return false;
    const auto expected = compute_stride(shape);
    return strides == expected;
}
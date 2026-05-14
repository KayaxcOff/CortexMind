//
// Created by muham on 13.05.2026.
//

#include "CortexMind/framework/Tools/tensor_meta.hpp"
#include <numeric>
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

BroadcastKind cortex::_fw::classify_broadcast(const std::vector<i64>& shape_x, const std::vector<i64>& shape_y) {
    if (!is_broadcastable(shape_x, shape_y)) {
        return BroadcastKind::None;
    }

    if (shape_x == shape_y) {
        return BroadcastKind::None;
    }

    const size_t rank_x = shape_x.size();
    const size_t rank_y = shape_y.size();

    if (rank_y == 1 && shape_y[0] == shape_x.back()) {
        return BroadcastKind::Row;
    }

    if (rank_x == 1 && shape_x[0] == shape_y.back()) {
        return BroadcastKind::Col;
    }

    return BroadcastKind::General;
}
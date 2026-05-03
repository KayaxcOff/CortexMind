//
// Created by muham on 18.04.2026.
//

#include "CortexMind/framework/Tools/tensor_utils.hpp"
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw;

std::vector<i64> cortex::_fw::compute_stride(const std::vector<i64> &shape) {
    std::vector<i64> output(shape.size());
    output[shape.size() - 1] = 1;
    for (i32 i = static_cast<i32>(shape.size()) - 2; i >= 0; --i) {
        output[i] = output[i + 1] * shape[i + 1];
    }
    return output;
}

size_t cortex::_fw::compute_numel(const std::vector<i64> &shape) {
    i64 output = 1;
    for (const auto item : shape) {
        output *= item;
    }
    return output;
}

i64 cortex::_fw::compute_offset(const std::vector<i64> &index, const std::vector<i64> &stride) {
    i64 output = 0;
    for (size_t i = 0; i < index.size(); ++i) {
        output += index[i] * stride[i];
    }
    return output;
}

bool cortex::_fw::is_contiguous(const std::vector<i64> &shape, const std::vector<i64> &stride) {
    if (shape.empty()) {
        return true;
    }

    i64 expected = 1;

    for (i32 i = static_cast<i32>(shape.size()) - 1; i >= 0; --i) {
        if (stride[i] != expected) {
            return false;
        }
        expected *= shape[i];
    }
    return true;
}

BroadcastInfo cortex::_fw::compute_broadcast(const std::vector<i64> &shape_x, const std::vector<i64> &shape_y) {
    const size_t ndim_x = shape_x.size();
    const size_t ndim_y = shape_y.size();
    const size_t ndim   = std::max(ndim_x, ndim_y);

    CXM_ASSERT(ndim <= CXM_MAX_DIMS,
        "compute_broadcast()", "Number of dimensions exceeds CXM_MAX_DIMS");

    BroadcastInfo info{};
    info.ndim = static_cast<i32>(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        const size_t ix = ndim - 1 - i;
        const size_t dx = (i < ndim_x) ? static_cast<size_t>(shape_x[ndim_x - 1 - i]) : 1;
        const size_t dy = (i < ndim_y) ? static_cast<size_t>(shape_y[ndim_y - 1 - i]) : 1;

        CXM_ASSERT(dx == dy || dx == 1 || dy == 1,
            "compute_broadcast()", "Shapes are not broadcastable");

        info.shape[ix] = std::max(dx, dy);
    }

    info.stride_z[ndim - 1] = 1;
    for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
        info.stride_z[i] = info.stride_z[i + 1] * info.shape[i + 1];
    }

    for (size_t i = 0; i < ndim; ++i) {
        const size_t ix = ndim - 1 - i;
        const size_t dx = (i < ndim_x) ? static_cast<size_t>(shape_x[ndim_x - 1 - i]) : 1;
        info.stride_x[ix] = (dx == 1 && info.shape[ix] != 1) ? 0 : info.stride_z[ix];
    }

    for (size_t i = 0; i < ndim; ++i) {
        const size_t ix = ndim - 1 - i;
        const size_t dy = (i < ndim_y) ? static_cast<size_t>(shape_y[ndim_y - 1 - i]) : 1;
        info.stride_y[ix] = (dy == 1 && info.shape[ix] != 1) ? 0 : info.stride_z[ix];
    }

    return info;
}

bool cortex::_fw::shapes_equal(const std::vector<i64> &shape_x, const std::vector<i64> &shape_y) {
    if (shape_x.size() != shape_y.size()) {
        return false;
    }
    for (size_t i = 0; i < shape_x.size(); ++i) {
        if (shape_x[i] != shape_y[i]) {
            return false;
        }
    }
    return true;
}

bool cortex::_fw::is_broadcastable(const std::vector<i64> &shape_x, const std::vector<i64> &shape_y) {
    const size_t ndim_x = shape_x.size();
    const size_t ndim_y = shape_y.size();
    const size_t ndim   = std::max(ndim_x, ndim_y);

    for (size_t i = 0; i < ndim; ++i) {
        const size_t dx = (i < ndim_x) ? static_cast<size_t>(shape_x[ndim_x - 1 - i]) : 1;
        const size_t dy = (i < ndim_y) ? static_cast<size_t>(shape_y[ndim_y - 1 - i]) : 1;
        if (dx != dy && dx != 1 && dy != 1) {
            return false;
        }
    }
    return true;
}

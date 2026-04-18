//
// Created by muham on 18.04.2026.
//

#include "CortexMind/framework/Tools/tensor_utils.hpp"

using namespace cortex::_fw;

std::vector<i64> cortex::_fw::compute_stride(const std::vector<i64> &shape) {
    std::vector<i64> output(shape.size());
    for (i32 i = static_cast<i32>(shape.size()) - 1; i >= 0; --i) {
        output[i] *= shape[i];
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

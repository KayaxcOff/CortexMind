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

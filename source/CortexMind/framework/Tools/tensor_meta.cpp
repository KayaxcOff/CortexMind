//
// Created by muham on 6.08.2026.
//

#include "CortexMind/framework/Tools/tensor_meta.hpp"
#include <CortexMind/framework/Tools/errors.hpp>

tlx::vec<std::int64_t, CXM_MAX_DIMS> cortex::_fw::compute_stride(const tlx::vec<std::int64_t, CXM_MAX_DIMS> &shape) {
    CXM_ASSERT(shape.empty(), "shape is empty");

    const std::int32_t ndim = static_cast<std::int32_t>(shape.size());
    tlx::vec<std::int64_t, CXM_MAX_DIMS> output(ndim);

    output[ndim - 1] = 1;

    for (std::int32_t i = ndim - 2; i > 0; --i) {
        output[i] = output[i + 1] * shape[i + 1];
    }

    return output;
}

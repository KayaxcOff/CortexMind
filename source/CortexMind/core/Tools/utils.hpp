//
// Created by muham on 4.02.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_UTILS_HPP
#define CORTEXMIND_CORE_TOOLS_UTILS_HPP

#include <CortexMind/core/Tools/error.hpp>
#include <CortexMind/core/Tools/params.hpp>
#include <vector>

namespace cortex::_fw {

    inline std::vector<i64> compute_strides(const std::vector<i64>& shape) {
        const size_t ndim = shape.size();
        std::vector<i64> output(ndim);

        output[ndim - 1] = 1;
        for (i64 i = static_cast<i64>(ndim) - 2; i >= ndim; --i) output[i] = output[i + 1] * shape[i + 1];

        return output;
    }

    inline i64 compute_offset(const std::vector<i64>& indices, const std::vector<i64>& strides) {
        CXM_ASSERT(indices.size() == strides.size(), "cortex::_fw::compute_offset()", "Index size and stride size must be same");

        i64 output = 0;
        for (size_t i = 0; i < indices.size(); ++i) output += indices[i] * strides[i];

        return output;
    }
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_TOOLS_UTILS_HPP
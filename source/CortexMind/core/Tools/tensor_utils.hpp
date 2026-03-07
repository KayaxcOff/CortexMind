//
// Created by muham on 22.02.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_TENSOR_UTILS_HPP
#define CORTEXMIND_CORE_TOOLS_TENSOR_UTILS_HPP

#include <CortexMind/core/Tools/err.hpp>
#include <CortexMind/core/Tools/warn.hpp>
#include <CortexMind/core/Tools/params.hpp>
#include <vector>
#include <ranges>
#include <algorithm>

namespace cortex::_fw {
    [[nodiscard]] inline
    std::vector<i64> compute_strides(const std::vector<i64>& shape) {
        CXM_ASSERT(!shape.empty(), "cortex::_fw::compute_strides()", "Shape cannot be empty.");

        const size_t ndim = shape.size();
        std::vector<i64> output(ndim);

        output[ndim - 1] = 1;
        for (i64 i = static_cast<i64>(ndim) - 2; i >= 0; --i) output[i] = output[i + 1] * shape[i + 1];

        return output;
    }

    [[nodiscard]] inline
    i64 compute_offset(const std::vector<i64>& indices, const std::vector<i64>& strides) {
        CXM_ASSERT(indices.size() == strides.size(), "cortex::_fw::compute_offset()", "Index size and stride size must be same.");

        i64 output = 0;
        for (size_t i = 0; i < indices.size(); ++i) output += indices[i] * strides[i];
        return output;
    }

    [[nodiscard]] inline
    i64 compute_numel(const std::vector<i64>& shape) {
        CXM_ASSERT(!shape.empty(), "cortex::_fw::compute_numel()", "Shape cannot be empty.");

        i64 n = 1;
        for (const i64 dim : shape) {
            CXM_WARN_IF(dim > 0, "cortex::_fw::compute_numel()", "Shape contains zero or negative dimension.");
            n *= dim;
        }
        return n;
    }

    [[nodiscard]] inline
    bool is_valid_shape(const std::vector<i64>& shape) {
        return !shape.empty() && std::ranges::all_of(shape, [](const i64 dim) { return dim > 0; });
    }

    [[nodiscard]] inline
    bool is_contiguous(const std::vector<i64>& shape, const std::vector<i64>& strides) {
        CXM_ASSERT(shape.size() == strides.size(), "cortex::_fw::is_contiguous()", "Shape and strides must have the same size.");

        const std::vector<i64> expected = compute_strides(shape);
        return strides == expected;
    }

    [[nodiscard]] inline
    bool is_broadcastable(const std::vector<i64>& a, const std::vector<i64>& b) {
        const size_t na = a.size();
        const size_t nb = b.size();
        const size_t n  = std::max(na, nb);

        for (size_t i = 0; i < n; ++i) {
            const i64 da = i < na ? a[na - 1 - i] : 1;
            const i64 db = i < nb ? b[nb - 1 - i] : 1;
            if (da != db && da != 1 && db != 1) return false;
        }
        return true;
    }

    [[nodiscard]] inline
    std::vector<i64> broadcast_shape(const std::vector<i64>& a, const std::vector<i64>& b) {
        CXM_ASSERT(is_broadcastable(a, b), "cortex::_fw::broadcast_shape()", "Shapes are not broadcastable.");

        const size_t na = a.size();
        const size_t nb = b.size();
        const size_t n  = std::max(na, nb);

        std::vector<i64> output(n);
        for (size_t i = 0; i < n; ++i) {
            const i64 da = i < na ? a[na - 1 - i] : 1;
            const i64 db = i < nb ? b[nb - 1 - i] : 1;
            output[n - 1 - i] = std::max(da, db);
        }
        return output;
    }
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_TOOLS_TENSOR_UTILS_HPP
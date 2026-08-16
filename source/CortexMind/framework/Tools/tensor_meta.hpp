//
// Created by muham on 6.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HP
#define CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HP

#include <CortexMind/framework/Shape/shape.hpp>
#include <CortexMind/runtime/macros.hpp>
#include <tlx/vector.hpp>
#include <vector>

namespace cortex::_fw {
    [[nodiscard]]
    tlx::vec<std::int64_t, CXM_MAX_DIMS> compute_stride(const tlx::vec<std::int64_t, CXM_MAX_DIMS> &shape);
    [[nodiscard]]
    std::size_t compute_size(const tlx::vec<std::int64_t, CXM_MAX_DIMS> &shape);
    [[nodiscard]]
    std::size_t compute_size(const std::vector<std::int64_t> &shape);
    [[nodiscard]]
    std::int64_t compute_idx(const tlx::vec<std::int64_t, CXM_MAX_DIMS> &indices, const TensorShape& shape);
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HP
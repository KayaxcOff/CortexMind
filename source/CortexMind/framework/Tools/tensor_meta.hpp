//
// Created by muham on 6.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HP
#define CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HP

#include <CortexMind/runtime/macros.hpp>
#include <tlx/vector.hpp>

namespace cortex::_fw {
    [[nodiscard]]
    tlx::vec<std::int64_t, CXM_MAX_DIMS> compute_stride(const tlx::vec<std::int64_t, CXM_MAX_DIMS> &shape);
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_TENSOR_META_HP
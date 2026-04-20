//
// Created by muham on 20.04.2026.
//

#ifndef CORTEXMIND_TOOLS_COMPARISON_OPERATIONS_HPP
#define CORTEXMIND_TOOLS_COMPARISON_OPERATIONS_HPP

#include <CortexMind/tools/params.hpp>

namespace cortex {
    [[nodiscard]]
    int32 argmax(const tensor& x);
    [[nodiscard]]
    int32 argmin(const tensor& x);
} //namespace cortex

#endif //CORTEXMIND_TOOLS_COMPARISON_OPERATIONS_HPP
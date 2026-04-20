//
// Created by muham on 20.04.2026.
//

#ifndef CORTEXMIND_TOOLS_ARITHMETIC_OPERATIONS_HPP
#define CORTEXMIND_TOOLS_ARITHMETIC_OPERATIONS_HPP

#include <CortexMind/tools/params.hpp>

namespace cortex {
    [[nodiscard]]
    tensor add(const tensor& Xx, const tensor& Xy);
    [[nodiscard]]
    tensor subtract(const tensor& Xx, const tensor& Xy);
    [[nodiscard]]
    tensor multiply(const tensor& Xx, const tensor& Xy);
    [[nodiscard]]
    tensor divide(const tensor& Xx, const tensor& Xy);
} //namespace cortex

#endif //CORTEXMIND_TOOLS_ARITHMETIC_OPERATIONS_HPP
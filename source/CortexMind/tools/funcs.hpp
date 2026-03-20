//
// Created by muham on 20.03.2026.
//

#ifndef CORTEXMIND_TOOLS_FUNCS_HPP
#define CORTEXMIND_TOOLS_FUNCS_HPP

#include <CortexMind/tools/params.hpp>

namespace cortex {
    [[nodiscard]]
    tensor addition(tensor& Xx, const tensor& Yy) noexcept;
    [[nodiscard]]
    tensor subtract(tensor& Xx, const tensor& Yy) noexcept;
    [[nodiscard]]
    tensor multiply(tensor& Xx, const tensor& Yy) noexcept;
    [[nodiscard]]
    tensor divide(tensor& Xx, const tensor& Yy) noexcept;
    [[nodiscard]]
    tensor max(const tensor& Xx, const tensor& Yy) noexcept;
    [[nodiscard]]
    tensor min(const tensor& Xx, const tensor& Yy) noexcept;
} // namespace cortex

#endif //CORTEXMIND_TOOLS_FUNCS_HPP
//
// Created by muham on 25.02.2026.
//

#ifndef CORTEXMIND_TOOLS_CAST_HPP
#define CORTEXMIND_TOOLS_CAST_HPP

#include <CortexMind/tools/params.hpp>
#include <CortexMind/tools/device.hpp>

namespace cortex {
    [[nodiscard]] inline
    tensor cast(const std::vector<_fw::i64>& shape, const dev _dev, const bool _requires_grad = false) {
        return {shape, _dev, _requires_grad};
    }
} // namespace cortex

#endif //CORTEXMIND_TOOLS_CAST_HPP
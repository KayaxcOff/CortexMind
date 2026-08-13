//
// Created by muham on 13.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_GRID_CUH
#define CORTEXMIND_FRAMEWORK_TOOLS_GRID_CUH

#include <cstdint>
#include <utility>

namespace cortex::_fw {
    constexpr std::int32_t kBlockSize = 256;

    [[nodiscard]]
    std::int32_t grid(std::size_t n, std::size_t byte);

    template<std::size_t byte>
    [[nodiscard]]
    std::int32_t grid(const std::size_t n) {
        const std::size_t vector_count = n / byte;

        return static_cast<std::int32_t>(
            (std::max<std::size_t>(vector_count, 1) + kBlockSize - 1) / kBlockSize
        );
    }
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_GRID_CUH
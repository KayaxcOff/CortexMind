//
// Created by muham on 13.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_GRID_CUH
#define CORTEXMIND_FRAMEWORK_TOOLS_GRID_CUH

#include <cstdint>
#include <utility>

namespace cortex::_fw {
    /**
     * @brief Number of threads launched in each CUDA block.
     */
    constexpr std::int32_t kBlockSize = 256;

    /**
     * @brief Calculates the number of CUDA blocks required for a workload.
     *
     * The input size is converted into the number of work units by dividing
     * @p n by @p byte. The resulting work-unit count is then rounded up to
     * the number of blocks required for a grid-stride kernel using
     * @ref kBlockSize threads per block.
     *
     * @param n Number of elements or bytes in the workload.
     * @param byte Size of a single work unit.
     *
     * @return Number of CUDA blocks required to process the workload.
     */
    [[nodiscard]]
    std::int32_t grid(std::size_t n, std::size_t byte);

    /**
     * @brief Calculates the number of CUDA blocks for a compile-time work-unit size.
     *
     * This overload performs the same calculation as grid(), but takes the
     * work-unit size as a compile-time template parameter. At least one CUDA
     * block is returned, even when the calculated number of work units is zero.
     *
     * @tparam byte Size of a single work unit.
     * @param n Number of elements or bytes in the workload.
     *
     * @return Number of CUDA blocks required to process the workload.
     */
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
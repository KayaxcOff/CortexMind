//
// Created by muham on 13.08.2026.
//

#include "CortexMind/framework/Tools/grid.cuh"

std::int32_t cortex::_fw::grid(const std::size_t n, const std::size_t byte) {
    return static_cast<std::int32_t>((n / byte + kBlockSize - 1) / kBlockSize);
}
//
// Created by muham on 29.03.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_UTILITIES_HPP
#define CORTEXMIND_CORE_TOOLS_UTILITIES_HPP

#include <CortexMind/core/Engine/CUDA/params.h>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda {
    inline constexpr i32 WARP_SIZE = 32;
    inline constexpr i32 BLOCK_SIZE_1D = 256;
    inline constexpr i32 BLOCK_SIZE_2D = 16;
    inline constexpr i32 MAT_TILE = 16;

    static constexpr f32 CXM_F32_MAX =  3.402823466e+38f;
    static constexpr f32 CXM_F32_MIN = -3.402823466e+38f;

    [[nodiscard]]
    __host__ inline dim3 grid1d(const size_t n, const i32 block = BLOCK_SIZE_1D) {
        return dim3(static_cast<unsigned>((n + block - 1) / block));
    }
    [[nodiscard]]
    __host__ inline dim3 grid2d(const size_t rows, const size_t cols, const i32 tile = BLOCK_SIZE_2D) {
        return dim3(static_cast<unsigned>((cols + tile - 1) / tile), static_cast<unsigned>((rows + tile - 1) / tile));
    }
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_TOOLS_UTILITIES_HPP
//
// Created by muham on 21.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX2_MATRIX_HPP
#define CORTEXMIND_CORE_ENGINE_AVX2_MATRIX_HPP

#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::avx2 {
    struct matrix_t {
        static void add(const f32* x, const f32* y, f32* z, size_t idx);
        static void sub(const f32* x, const f32* y, f32* z, size_t idx);
        static void mul(const f32* x, const f32* y, f32* z, size_t idx);
        static void div(const f32* x, const f32* y, f32* z, size_t idx);
        static void fma(const f32* x, const f32* y, const f32* z, f32* m, size_t idx);
        static void matmul(const f32* x, const f32* y, f32* z, size_t xIdx, size_t yIdx, size_t zIdx);
    };
} // namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX2_MATRIX_HPP
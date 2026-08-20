//
// Created by muham on 7.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_WISE_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_WISE_HPP

#include <CortexMind/framework/Tools/view.hpp>
#include <cstdint>
#include <cwchar>

namespace cortex::_fw::avx2 {
    /**
     * @brief Element-wise mathematical operations for contiguous float arrays.
     *
     * Provides vectorized mathematical transformations over contiguous
     * single-precision floating-point arrays.
     *
     * Operations are implemented using the CortexMind AVX2 primitives for
     * the vectorized portion of the input and scalar operations for any
     * remaining tail elements.
     *
     * Unless otherwise specified, operations write their results to the
     * destination buffer without modifying the source buffer.
     */
    struct wise {
        static void square(const TensorView& Xx, TensorView& Xz);
        static void pow(const TensorView& Xx, float value, TensorView& Xz);
        static void pow(const TensorView& Xx, const TensorView& Xy, TensorView& Xz);
        static void sqrt(const TensorView& Xx, TensorView& Xz);
        static void rsqrt(const TensorView& Xx, TensorView& Xz);
        static void log(const TensorView& Xx, TensorView& Xz);
        static void exp(const TensorView& Xx, TensorView& Xz);
        static void erf(const TensorView& Xx, TensorView& Xz);
        static void sin(const TensorView& Xx, TensorView& Xz);
        static void cos(const TensorView& Xx, TensorView& Xz);
        static void abs(const TensorView& Xx, TensorView& Xz);
        static void neg(const TensorView& Xx, TensorView& Xz);
        static void rcp(const TensorView& Xx, TensorView& Xz);
        static void inverse(const TensorView& Xx, TensorView& Xz);
        static void sign(const TensorView& Xx, TensorView& Xz);

        static void lerp(const float* __restrict Xx, float value1, float value2, float* __restrict Xz, std::size_t N);
        static void clamp(const float* __restrict Xx, float min, float max, float* __restrict Xz, std::size_t N);
        static void gather(const float* __restrict Xx, const std::int32_t* __restrict Xy, float* __restrict Xz, std::size_t N);
        static void gather(const std::int32_t* __restrict Xx, const std::int32_t* __restrict Xy, std::int32_t* __restrict Xz, std::size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_WISE_HPP
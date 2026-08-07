//
// Created by muham on 7.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_WISE_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_WISE_HPP

#include <cwchar>

namespace cortex::_fw::avx2 {
    struct wise {
        static void square(const float* __restrict Xx, float* __restrict Xz, std::size_t n);
        static void pow(const float* __restrict Xx, float value, float* __restrict Xz, std::size_t n);
        static void pow(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t n);
        static void sqrt(const float* __restrict Xx, float* __restrict Xz, std::size_t n);
        static void rsqrt(const float* __restrict Xx, float* __restrict Xz, std::size_t n);
        static void log(const float* __restrict Xx, float* __restrict Xz, std::size_t n);
        static void exp(const float* __restrict Xx, float* __restrict Xz, std::size_t n);
        static void erf(const float* __restrict Xx, float* __restrict Xz, std::size_t n);
        static void sin(const float* __restrict Xx, float* __restrict Xz, std::size_t n);
        static void cos(const float* __restrict Xx, float* __restrict Xz, std::size_t n);
        static void abs(const float* __restrict Xx, float* __restrict Xz, std::size_t n);
        static void neg(const float* __restrict Xx, float* __restrict Xz, std::size_t n);
        static void rcp(const float* __restrict Xx, float* __restrict Xz, std::size_t n);
        static void inverse(const float* __restrict Xx, float* __restrict Xz, std::size_t n);
        static void lerp(const float* __restrict Xx, float value1, float value2, float* __restrict Xz, std::size_t n);
        static void clamp(const float* __restrict Xx, float min, float max, float* __restrict Xz, std::size_t n);
        static void sign(const float* __restrict Xx, float* __restrict Xz, std::size_t n);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_WISE_HPP
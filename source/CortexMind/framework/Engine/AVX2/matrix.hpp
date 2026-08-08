//
// Created by muham on 8.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MATRIX_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MATRIX_HPP

#include <cwchar>

namespace cortex::_fw::avx2 {
    struct matrix_t {
        static void add(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t N);
        static void sub(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t N);
        static void mul(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t N);
        static void div(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t N);

        static void max(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t N);
        static void min(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t N);

        static void matmul(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t xN, std::size_t yN, std::size_t zN);

        static void add(float* Xx, const float* __restrict Xy, std::size_t N);
        static void sub(float* Xx, const float* __restrict Xy, std::size_t N);
        static void mul(float* Xx, const float* __restrict Xy, std::size_t N);
        static void div(float* Xx, const float* __restrict Xy, std::size_t N);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_MATRIX_HPP
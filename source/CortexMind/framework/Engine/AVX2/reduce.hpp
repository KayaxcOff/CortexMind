//
// Created by muham on 8.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_REDUCE_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_REDUCE_HPP

#include <cstdint>
#include <cwchar>

namespace cortex::_fw::avx2 {
    struct reduce {
        static void sum(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        static void sum(const float* __restrict Xx, float* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        static void mean(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        static void mean(const float* __restrict Xx, float* __restrict Xy, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        static void var(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        static void var(const float* __restrict Xx, float* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        static void stdv(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        static void stdv(const float* __restrict Xx, float* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        static void max(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        static void max(const float* __restrict Xx, float* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        static void min(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        static void min(const float* __restrict Xx, float* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        static void l1(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        static void l1(const float* __restrict, float* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        static void l2(const float* __restrict Xx, float* __restrict Xz, std::size_t N);
        static void l2(const float* __restrict, float* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        static void argmax(const float* __restrict Xx, std::int64_t* __restrict Xz, std::size_t N);
        static void argmax(const float* __restrict Xx, std::int64_t* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
        static void argmin(const float* __restrict Xx, std::int64_t* __restrict Xz, std::size_t N);
        static void argmin(const float* __restrict Xx, std::int64_t* __restrict Xz, std::size_t outer_size, std::size_t dim_size, std::size_t inner_size);
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_REDUCE_HPP
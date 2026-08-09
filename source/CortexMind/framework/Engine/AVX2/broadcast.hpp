//
// Created by muham on 9.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_BROADCAST_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_BROADCAST_HPP

#include <CortexMind/framework/Tools/broadcast_info.hpp>
#include <cwchar>

namespace cortex::_fw::avx2 {
    struct Broadcast {
        struct row {
            static void add(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t M, std::size_t N);
            static void sub(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t M, std::size_t N);
            static void mul(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t M, std::size_t N);
            static void div(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t M, std::size_t N);

            static void add(float* Xx, const float* __restrict Xy, std::size_t M, std::size_t N);
            static void sub(float* Xx, const float* __restrict Xy, std::size_t M, std::size_t N);
            static void mul(float* Xx, const float* __restrict Xy, std::size_t M, std::size_t N);
            static void div(float* Xx, const float* __restrict Xy, std::size_t M, std::size_t N);
        };

        struct col {
            static void add(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t M, std::size_t N);
            static void sub(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t M, std::size_t N);
            static void mul(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t M, std::size_t N);
            static void div(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, std::size_t M, std::size_t N);

            static void add(float* Xx, const float* __restrict Xy, std::size_t M, std::size_t N);
            static void sub(float* Xx, const float* __restrict Xy, std::size_t M, std::size_t N);
            static void mul(float* Xx, const float* __restrict Xy, std::size_t M, std::size_t N);
            static void div(float* Xx, const float* __restrict Xy, std::size_t M, std::size_t N);
        };

        struct general {
            static void add(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, const BroadcastInfo& info);
            static void sub(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, const BroadcastInfo& info);
            static void mul(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, const BroadcastInfo& info);
            static void div(const float* __restrict Xx, const float* __restrict Xy, float* __restrict Xz, const BroadcastInfo& info);
        };
    };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_BROADCAST_HPP
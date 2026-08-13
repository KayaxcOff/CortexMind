//
// Created by muham on 13.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_SCALAR_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_SCALAR_CUH

#include <tlx/types.hpp>

namespace cortex::_fw::nv {
    struct ScalarKernel {
        static void add(const float* __restrict Xx, float value, float* __restrict Xz, std::size_t N);
        static void add(const tlx::bfloat16* __restrict Xx, tlx::bfloat16 value, tlx::bfloat16* __restrict Xz, std::size_t N);
        static void add(const tlx::half* __restrict Xx, tlx::half value, tlx::half* __restrict Xz, std::size_t N);

        static void sub(const float* __restrict Xx, float value, float* __restrict Xz, std::size_t N);
        static void sub(const tlx::bfloat16* __restrict Xx, tlx::bfloat16 value, tlx::bfloat16* __restrict Xz, std::size_t N);
        static void sub(const tlx::half* __restrict Xx, tlx::half value, tlx::half* __restrict Xz, std::size_t N);

        static void mul(const float* __restrict Xx, float value, float* __restrict Xz, std::size_t N);
        static void mul(const tlx::bfloat16* __restrict Xx, tlx::bfloat16 value, tlx::bfloat16* __restrict Xz, std::size_t N);
        static void mul(const tlx::half* __restrict Xx, tlx::half value, tlx::half* __restrict Xz, std::size_t N);

        static void div(const float* __restrict Xx, float value, float* __restrict Xz, std::size_t N);
        static void div(const tlx::bfloat16* __restrict Xx, tlx::bfloat16 value, tlx::bfloat16* __restrict Xz, std::size_t N);
        static void div(const tlx::half* __restrict Xx, tlx::half value, tlx::half* __restrict Xz, std::size_t N);

        static void add(float* Xx, float value, std::size_t N);
        static void add(tlx::bfloat16* Xx, tlx::bfloat16 value, std::size_t N);
        static void add(tlx::half* Xx, tlx::half value, std::size_t N);

        static void sub(float* Xx, float value, std::size_t N);
        static void sub(tlx::bfloat16* Xx, tlx::bfloat16 value, std::size_t N);
        static void sub(tlx::half* Xx, tlx::half value, std::size_t N);

        static void mul(float* Xx, float value, std::size_t N);
        static void mul(tlx::bfloat16* Xx, tlx::bfloat16 value, std::size_t N);
        static void mul(tlx::half* Xx, tlx::half value, std::size_t N);

        static void div(float* Xx, float value, std::size_t N);
        static void div(tlx::bfloat16* Xx, tlx::bfloat16 value, std::size_t N);
        static void div(tlx::half* Xx, tlx::half value, std::size_t N);
    };
} //namespace cortex::_fw::nv

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_SCALAR_CUH
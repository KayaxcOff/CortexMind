//
// Created by muham on 17.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_MATRIX_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_MATRIX_CUH

#include <tlx/concepts.hpp>

namespace cortex::_fw::nv {
    struct Matrix {
        template<tlx::float_like T>
        static void add(const T* __restrict Xx, const T* __restrict Xy, T* __restrict Xz, std::size_t N);
        template<tlx::float_like T>
        static void sub(const T* __restrict Xx, const T* __restrict Xy, T* __restrict Xz, std::size_t N);
        template<tlx::float_like T>
        static void mul(const T* __restrict Xx, const T* __restrict Xy, T* __restrict Xz, std::size_t N);
        template<tlx::float_like T>
        static void div(const T* __restrict Xx, const T* __restrict Xy, T* __restrict Xz, std::size_t N);
        template<tlx::float_like T>
        static void max(const T* __restrict Xx, const T* __restrict Xy, T* __restrict Xz, std::size_t N);
        template<tlx::float_like T>
        static void min(const T* __restrict Xx, const T* __restrict Xy, T* __restrict Xz, std::size_t N);

        template<tlx::float_like T>
        static void matmul(const T* __restrict Xx, const T* __restrict Xy, T* __restrict Xz, std::size_t xN, std::size_t yN, std::size_t zN);

        template<tlx::float_like T>
        static void add(T* Xx, const T* __restrict Xy, std::size_t N);
        template<tlx::float_like T>
        static void sub(T* Xx, const T* __restrict Xy, std::size_t N);
        template<tlx::float_like T>
        static void mul(T* Xx, const T* __restrict Xy, std::size_t N);
        template<tlx::float_like T>
        static void div(T* Xx, const T* __restrict Xy, std::size_t N);
    };
} //namespace cortex::_fw::nv

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_MATRIX_CUH
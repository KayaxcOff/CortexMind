//
// Created by muham on 14.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_ELEMENT_WISE_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_ELEMENT_WISE_CUH

#include <tlx/concepts.hpp>

namespace cortex::_fw::nv {
    struct ElementWise {
        template<tlx::float_like T>
        static void square(const T* __restrict Xx, T* __restrict Xz, std::size_t N);
        template<tlx::float_like T>
        static void pow(const T* __restrict Xx, T value, T* __restrict Xz, std::size_t N);
        template<tlx::float_like T>
        static void pow(const T* __restrict Xx, const T* __restrict Xy, T* __restrict Xz, std::size_t N);
        template<tlx::float_like T>
        static void sqrt(const T* __restrict Xx, T* __restrict Xz, std::size_t N);
        template<tlx::float_like T>
        static void rsqrt(const T* __restrict Xx, T* __restrict Xz, std::size_t N);
        template<tlx::float_like T>
        static void log(const T* __restrict Xx, T* __restrict Xz, std::size_t N);
        template<tlx::float_like T>
        static void exp(const T* __restrict Xx, T* __restrict Xz, std::size_t N);
        template<tlx::float_like T>
        static void erf(const T* __restrict Xx, T* __restrict Xz, std::size_t N);
    };
} //namespace cortex::_fw::nv

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_ELEMENT_WISE_CUH
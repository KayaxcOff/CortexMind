//
// Created by muham on 19.04.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_CUDA_ELEM_WISE_H
#define CORTEXMIND_CORE_ENGINE_CUDA_ELEM_WISE_H

#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::cuda {
    struct ElementWise {
        static void pow(const f32* __restrict Xx, f32 exp, f32* __restrict Xz, size_t N);
        static void sqrt(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void square(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void log(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
        static void exp(const f32* __restrict Xx, f32* __restrict Xz, size_t N);
    };
} //namespace cortex::_fw::cuda

#endif //CORTEXMIND_CORE_ENGINE_CUDA_ELEM_WISE_H
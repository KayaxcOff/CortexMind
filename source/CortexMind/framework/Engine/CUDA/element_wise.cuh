//
// Created by muham on 14.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_ELEMENT_WISE_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_ELEMENT_WISE_CUH

#include <CortexMind/framework/Tools/view.hpp>

namespace cortex::_fw::nv {
    struct ElementWise {
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
    };
} //namespace cortex::_fw::nv

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_ELEMENT_WISE_CUH
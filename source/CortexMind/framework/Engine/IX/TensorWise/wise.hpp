//
// Created by muham on 5.06.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_IX_TENSOR_WISE_WISE_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_IX_TENSOR_WISE_WISE_HPP

#include <CortexMind/framework/Storage/stor.hpp>

namespace cortex::_fw::ix {
    struct TensorWise {
        static void pow(const TensorStorage* __restrict Xx, f32 exp, TensorStorage* __restrict Xz);
        static void sqrt(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void rsqrt(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void square(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void exp(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void exp2(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void exp10(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void log(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void log2(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void log10(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void erf(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void sin(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void cos(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void tan(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void cot(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void abs(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void neg(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void sign(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void reciprocal(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz);
        static void clamp(const TensorStorage* __restrict Xx, f32 min, f32 max, TensorStorage* __restrict Xz);
    };
} //namespace cortex::_fw::ix

#endif //CORTEXMIND_FRAMEWORK_ENGINE_IX_TENSOR_WISE_WISE_HPP
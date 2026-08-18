//
// Created by muham on 17.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_CUDA_MATRIX_CUH
#define CORTEXMIND_FRAMEWORK_ENGINE_CUDA_MATRIX_CUH

#include <CortexMind/framework/Tools/view.hpp>

namespace cortex::_fw::nv {
    struct Matrix {
        static void add(const TensorView& Xx, const TensorView& Xy, TensorView& Xz);
        static void sub(const TensorView& Xx, const TensorView& Xy, TensorView& Xz);
        static void mul(const TensorView& Xx, const TensorView& Xy, TensorView& Xz);
        static void div(const TensorView& Xx, const TensorView& Xy, TensorView& Xz);

        static void max(const TensorView& Xx, const TensorView& Xy, TensorView& Xz);
        static void min(const TensorView& Xx, const TensorView& Xy, TensorView& Xz);

        static void matmul(const TensorView& Xx, const TensorView& Xy, TensorView& Xz);

        static void add(TensorView& Xx, const TensorView& Xy);
        static void sub(TensorView& Xx, const TensorView& Xy);
        static void mul(TensorView& Xx, const TensorView& Xy);
        static void div(TensorView& Xx, const TensorView& Xy);
    };
} //namespace cortex::_fw::nv

#endif //CORTEXMIND_FRAMEWORK_ENGINE_CUDA_MATRIX_CUH
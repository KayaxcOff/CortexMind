//
// Created by muham on 6.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_AVX2_SCALAR_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_AVX2_SCALAR_HPP

#include <CortexMind/framework/Tools/view.hpp>

namespace cortex::_fw::avx2 {
   struct ScalarOp {
      static void add(const TensorView& Xx, float value, TensorView& Xz);
      static void sub(const TensorView& Xx, float value, TensorView& Xz);
      static void mul(const TensorView& Xx, float value, TensorView& Xz);
      static void div(const TensorView& Xx, float value, TensorView& Xz);

      static void add(TensorView& Xx, float value);
      static void sub(TensorView& Xx, float value);
      static void mul(TensorView& Xx, float value);
      static void div(TensorView& Xx, float value);
   };
} //namespace cortex::_fw::avx2

#endif //CORTEXMIND_FRAMEWORK_ENGINE_AVX2_SCALAR_HPP
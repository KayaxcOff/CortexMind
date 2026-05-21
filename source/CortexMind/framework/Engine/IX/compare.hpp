//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_IX_COMPARE_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_IX_COMPARE_HPP

#include <CortexMind/framework/Storage/stor.hpp>

namespace cortex::_fw::ix {
    struct CompareTo {
        static void greater(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz, size_t N);
        static void less(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz, size_t N);
        static void greater_eq(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz, size_t N);
        static void less_eq(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz, size_t N);
        static bool equal(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, size_t N);
    };
} //namespace cortex::_fw::ix

#endif //CORTEXMIND_FRAMEWORK_ENGINE_IX_COMPARE_HPP
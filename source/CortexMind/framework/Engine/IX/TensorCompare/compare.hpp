//
// Created by muham on 7.06.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_IX_TENSOR_COMPARE_COMPARE_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_IX_TENSOR_COMPARE_COMPARE_HPP

#include <CortexMind/framework/Storage/stor.hpp>

namespace cortex::_fw::ix {
    struct TensorCompare {
        static void gt(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz) noexcept;
        static void lt(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz) noexcept;
        static void eq(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz) noexcept;
        static void ge(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz) noexcept;
        static void le(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz) noexcept;
        static void neq(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz) noexcept;
    };
} // namespace cortex::_fw::ix

#endif //CORTEXMIND_FRAMEWORK_ENGINE_IX_TENSOR_COMPARE_COMPARE_HPP
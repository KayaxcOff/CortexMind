//
// Created by muham on 6.06.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_IX_TENSOR_OP_OP_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_IX_TENSOR_OP_OP_HPP

#include <CortexMind/framework/Shape/shape.hpp>
#include <CortexMind/framework/Storage/stor.hpp>

namespace cortex::_fw::ix {
    struct TensorOp {
        static void add(const TensorStorage* __restrict Xx, const TensorShape& shape_x, const TensorStorage* __restrict Xy, const TensorShape& shape_y, TensorStorage* __restrict Xz, const TensorShape& shape_z);
        static void sub(const TensorStorage* __restrict Xx, const TensorShape& shape_x, const TensorStorage* __restrict Xy, const TensorShape& shape_y, TensorStorage* __restrict Xz, const TensorShape& shape_z);
        static void mul(const TensorStorage* __restrict Xx, const TensorShape& shape_x, const TensorStorage* __restrict Xy, const TensorShape& shape_y, TensorStorage* __restrict Xz, const TensorShape& shape_z);
        static void div(const TensorStorage* __restrict Xx, const TensorShape& shape_x, const TensorStorage* __restrict Xy, const TensorShape& shape_y, TensorStorage* __restrict Xz, const TensorShape& shape_z);

        static void add(TensorStorage* Xx, const TensorShape& shape_x, const TensorStorage* __restrict Xy, const TensorShape& shape_y);
        static void sub(TensorStorage* Xx, const TensorShape& shape_x, const TensorStorage* __restrict Xy, const TensorShape& shape_y);
        static void mul(TensorStorage* Xx, const TensorShape& shape_x, const TensorStorage* __restrict Xy, const TensorShape& shape_y);
        static void div(TensorStorage* Xx, const TensorShape& shape_x, const TensorStorage* __restrict Xy, const TensorShape& shape_y);

        static void matmul(const TensorStorage* __restrict Xx, const TensorShape& shape_x, const TensorStorage* __restrict Xy, const TensorShape& shape_y, TensorStorage* __restrict Xz, const TensorShape& shape_z);
    };
} //namespace cortex::_fw::ix

#endif //CORTEXMIND_FRAMEWORK_ENGINE_IX_TENSOR_OP_OP_HPP
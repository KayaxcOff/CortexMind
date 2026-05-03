//
// Created by muham on 3.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_DISPATCH_MATRIX_HPP
#define CORTEXMIND_FRAMEWORK_DISPATCH_MATRIX_HPP

#include <CortexMind/framework/Memory/device.hpp>
#include <CortexMind/framework/Storage/stor.hpp>
#include <CortexMind/framework/Tools/params.hpp>
#include <CortexMind/core/Tools/broadcast.hpp>

namespace cortex::_fw::txl {
    class MatrixExecutor {
    public:
        explicit MatrixExecutor(sys::deviceType d_type = sys::deviceType::host);
        ~MatrixExecutor();

        void SetDevice(sys::deviceType _d_type);

        void addition(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz) const;
        void subtraction(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz) const;
        void multiply(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz) const;
        void division(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz) const;

        void addition(TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy) const;
        void subtraction(TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy) const;
        void multiply(TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy) const;
        void division(TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy) const;

        void matmul(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, TensorStorage* __restrict Xz, i64 M, i64 K, i64 N) const;

    private:
        sys::deviceType d_type;

        void dispatch_binary(const TensorStorage* Xx, const TensorStorage* Xy, TensorStorage* Xz,
                             // AVX2
                             void(*avx2_fn)(const f32*, const f32*, f32*, size_t),
                             void(*avx2_broadcast_fn)(const f32*, const f32*, f32*, const BroadcastInfo&),
                             // STD
                             void(*stl_fn)(const f32*, const f32*, f32*, size_t),
                             void(*stl_broadcast_fn)(const f32*, const f32*, f32*, const BroadcastInfo&)
                             #if CXM_IS_CUDA_AVAILABLE
                             , void(*cuda_fn)(const f32*, const f32*, f32*, size_t)
                             , void(*cuda_broadcast_fn)(const f32*, const f32*, f32*, size_t, const BroadcastInfo&)
                             #endif //#if CXM_IS_CUDA_AVAILABLE
                             ) const;
    };
} //namespace cortex::_fw::txl

#endif //CORTEXMIND_FRAMEWORK_DISPATCH_MATRIX_HPP
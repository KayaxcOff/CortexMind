//
// Created by muham on 29.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_DISPATCH_SCALAR_HPP
#define CORTEXMIND_FRAMEWORK_DISPATCH_SCALAR_HPP

#include <CortexMind/framework/Memory/device.hpp>
#include <CortexMind/framework/Storage/stor.hpp>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::txl {
    class TensorScalarExecutor {
    public:
        TensorScalarExecutor();
        ~TensorScalarExecutor();

        void SetDevice(sys::deviceType _d_type);

        void addition(const TensorStorage* __restrict Xx, f32 value, TensorStorage* __restrict Xz) const;
        void subtraction(const TensorStorage* __restrict Xx, f32 value, TensorStorage* __restrict Xz) const;
        void multiply(const TensorStorage* __restrict Xx, f32 value, TensorStorage* __restrict Xz) const;
        void division(const TensorStorage* __restrict Xx, f32 value, TensorStorage* __restrict Xz) const;

        void addition(TensorStorage* Xx, f32 value) const;
        void subtraction(TensorStorage* Xx, f32 value) const;
        void multiply(TensorStorage* Xx, f32 value) const;
        void division(TensorStorage* Xx, f32 value) const;

    private:
        sys::deviceType d_type;

        i32 max_dim;
    };
} //namespace cortex::_fw::txl

#endif //CORTEXMIND_FRAMEWORK_DISPATCH_SCALAR_HPP
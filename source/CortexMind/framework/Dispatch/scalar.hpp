//
// Created by muham on 29.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_DISPATCH_SCALAR_HPP
#define CORTEXMIND_FRAMEWORK_DISPATCH_SCALAR_HPP

#include <CortexMind/framework/Memory/device.hpp>
#include <CortexMind/framework/Storage/stor.hpp>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::txl {
    class TensorScalar {
    public:
        explicit TensorScalar(sys::deviceType _d_type);
        ~TensorScalar();

        void SetDevice(sys::deviceType _d_type);

        void add(const TensorStorage* __restrict Xx, f32 value, TensorStorage* __restrict Xz, size_t N) const;
        void sub(const TensorStorage* __restrict Xx, f32 value, TensorStorage* __restrict Xz, size_t N) const;
        void mul(const TensorStorage* __restrict Xx, f32 value, TensorStorage* __restrict Xz, size_t N) const;
        void div(const TensorStorage* __restrict Xx, f32 value, TensorStorage* __restrict Xz, size_t N) const;

        void add(TensorStorage* Xx, f32 value, size_t N) const;
        void sub(TensorStorage* Xx, f32 value, size_t N) const;
        void mul(TensorStorage* Xx, f32 value, size_t N) const;
        void div(TensorStorage* Xx, f32 value, size_t N) const;
    private:
        sys::deviceType d_type;
        i32 max_dim;
    };
} //namespace cortex::_fw::txl

#endif //CORTEXMIND_FRAMEWORK_DISPATCH_SCALAR_HPP
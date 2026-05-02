//
// Created by muham on 1.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_DISPATCH_REDUCE_HPP
#define CORTEXMIND_FRAMEWORK_DISPATCH_REDUCE_HPP

#include <CortexMind/framework/Memory/device.hpp>
#include <CortexMind/framework/Storage/stor.hpp>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::txl {
    class ReductionOps {
    public:
        explicit ReductionOps(sys::deviceType _d_type);
        ~ReductionOps();

        void SetDevice(sys::deviceType _d_type);

        [[nodiscard]]
        f32 sum(const TensorStorage* __restrict Xx, size_t N) const;
        [[nodiscard]]
        f32 mean(const TensorStorage* __restrict Xx, size_t N) const;
        [[nodiscard]]
        f32 var(const TensorStorage* __restrict Xx, size_t N) const;
        [[nodiscard]]
        f32 stdv(const TensorStorage* __restrict Xx, size_t N) const;
        [[nodiscard]]
        f32 min(const TensorStorage* __restrict Xx, size_t N) const;
        [[nodiscard]]
        f32 max(const TensorStorage* __restrict Xx, size_t N) const;
        [[nodiscard]]
        f32 norm1(const TensorStorage* __restrict Xx, size_t N) const;
        [[nodiscard]]
        f32 norm2(const TensorStorage* __restrict Xx, size_t N) const;
        [[nodiscard]]
        f32 dot(const TensorStorage* __restrict Xx, const TensorStorage* __restrict Xy, size_t N) const;

    private:
        sys::deviceType d_type;
        i32 max_dim;
    };
} //namespace cortex::_fw::txl

#endif //CORTEXMIND_FRAMEWORK_DISPATCH_REDUCE_HPP
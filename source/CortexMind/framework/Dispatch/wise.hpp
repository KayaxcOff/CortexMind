//
// Created by muham on 2.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_DISPATCH_WISE_HPP
#define CORTEXMIND_FRAMEWORK_DISPATCH_WISE_HPP

#include <CortexMind/framework/Tools/params.hpp>
#include <CortexMind/framework/Storage/stor.hpp>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::txl {
    class Wise {
    public:
        explicit Wise(sys::deviceType _d_type);
        ~Wise();

        void pow(const TensorStorage* __restrict Xx, f32 exp, TensorStorage* __restrict Xz, size_t N) const;
        void sqrt(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N) const;
        void log(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N) const;
        void exp(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz, size_t N) const;
    private:
        sys::deviceType d_type;
        i32 max_dim;
    };
} //namespace cortex::_fw::txl

#endif //CORTEXMIND_FRAMEWORK_DISPATCH_WISE_HPP
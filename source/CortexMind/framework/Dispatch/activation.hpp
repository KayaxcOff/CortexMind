//
// Created by muham on 1.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_DISPATCH_ACTIVATION_HPP
#define CORTEXMIND_FRAMEWORK_DISPATCH_ACTIVATION_HPP

#include <CortexMind/framework/Memory/device.hpp>
#include <CortexMind/framework/Storage/stor.hpp>
#include <CortexMind/framework/Tools/params.hpp>

namespace cortex::_fw::txl {
    class ActivationManager {
    public:
        ActivationManager();
        ~ActivationManager();

        void SetDevice(sys::deviceType _d_type);

        void ReLU(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz) const;
        void LeakyReLU(const TensorStorage* __restrict Xx, f32 alpha, TensorStorage* __restrict Xz) const;
        void Sigmoid(const TensorStorage* __restrict Xx, TensorStorage* __restrict Xz) const;
    private:
        sys::deviceType d_type;
        i32 max_dim;
    };
} //namespace cortex::_fw::txl

#endif //CORTEXMIND_FRAMEWORK_DISPATCH_ACTIVATION_HPP
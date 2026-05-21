//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_IX_ACTIVATION_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_IX_ACTIVATION_HPP

#include <CortexMind/framework/Memory/device_type.hpp>
#include <CortexMind/framework/Tools/err.hpp>

namespace cortex::_fw::ix {
    struct Activation {
        static void relu(const f32* __restrict Xx, f32* __restrict Xz, size_t N, sys::DeviceType device);
    };
} //namespace cortex::_fw::ix

#endif //CORTEXMIND_FRAMEWORK_ENGINE_IX_ACTIVATION_HPP
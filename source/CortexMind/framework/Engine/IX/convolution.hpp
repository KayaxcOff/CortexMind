//
// Created by muham on 24.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_IX_CONVOLUTION_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_IX_CONVOLUTION_HPP

#include <CortexMind/framework/Memory/device_type.hpp>
#include <CortexMind/framework/Tools/types.hpp>

namespace cortex::_fw::ix {
    struct Convolution {
        static void unfold(const f32* input, f32* col, i64 N, i64 C, i64 H, i64 W, i64 kH, i64 kW, i64 sH, i64 sW, i64 pH, i64 pW, i64 oH, i64 oW, sys::DeviceType device);
        static void fold(const f32* col, f32* input_grad, i64 N, i64 C, i64 H, i64 W, i64 kH, i64 kW, i64 sH, i64 sW, i64 pH, i64 pW, i64 oH, i64 oW, sys::DeviceType device);
    };
} //namespace cortex::_fw::ix

#endif //CORTEXMIND_FRAMEWORK_ENGINE_IX_CONVOLUTION_HPP
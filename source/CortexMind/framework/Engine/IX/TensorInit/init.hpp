//
// Created by muham on 5.06.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_ENGINE_IX_TENSOR_INIT_INIT_HPP
#define CORTEXMIND_FRAMEWORK_ENGINE_IX_TENSOR_INIT_INIT_HPP

#include <CortexMind/framework/Storage/stor.hpp>

namespace cortex::_fw::ix {
    struct TensorInit {
        static void rand(TensorStorage* __restrict x);
        static void uniform(TensorStorage* __restrict x, f32 min, f32 max);
        static void fill(TensorStorage* __restrict x, f32 value);
    };
} //namespace cortex::_fw::ix

#endif //CORTEXMIND_FRAMEWORK_ENGINE_IX_TENSOR_INIT_INIT_HPP
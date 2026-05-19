//
// Created by muham on 19.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_GRADIENT_PACK_HPP
#define CORTEXMIND_FRAMEWORK_GRADIENT_PACK_HPP

#include <CortexMind/framework/Tools/types.hpp>
#include <memory>
#include <vector>

namespace cortex::_fw {
    class Tensor;
    struct TensorStorage;
} //namespace cortex::_fw

namespace cortex::_fw::meta {
    struct GradientPacked {
        std::shared_ptr<TensorStorage> stor;
        std::shared_ptr<Tensor> gradient;
        std::vector<i64> shape;
        bool has_gradient;
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRADIENT_PACK_HPP
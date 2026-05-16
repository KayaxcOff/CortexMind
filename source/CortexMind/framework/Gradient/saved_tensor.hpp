//
// Created by muham on 16.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_GRADIENT_SAVED_TENSOR_HPP
#define CORTEXMIND_FRAMEWORK_GRADIENT_SAVED_TENSOR_HPP

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
        std::vector<i64> shape;
        std::shared_ptr<Tensor> gradient;
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRADIENT_SAVED_TENSOR_HPP
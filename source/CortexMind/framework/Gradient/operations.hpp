//
// Created by muham on 20.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_GRADIENT_OPERATIONS_HPP
#define CORTEXMIND_FRAMEWORK_GRADIENT_OPERATIONS_HPP

#include <CortexMind/framework/Gradient/flow.hpp>
#include <CortexMind/framework/Storage/stor.hpp>
#include <memory>

namespace cortex::_fw::meta {
    struct addition : GradientFlow {
        addition(const std::shared_ptr<TensorStorage> &tx_s, const std::shared_ptr<TensorStorage> &ty_s);

        void backward(MindTensor *_grad) override;
    private:
        std::weak_ptr<MindTensor> tx;
        std::weak_ptr<MindTensor> ty;
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRADIENT_OPERATIONS_HPP
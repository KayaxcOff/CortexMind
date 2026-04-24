//
// Created by muham on 20.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_GRADIENT_OPERATIONS_HPP
#define CORTEXMIND_FRAMEWORK_GRADIENT_OPERATIONS_HPP

#include <CortexMind/framework/Gradient/flow.hpp>
#include <memory>

namespace cortex::_fw {
    struct TensorStorage;
} //namespace cortex::_fw

namespace cortex::_fw::meta {
    struct addition : GradientFlow {
        addition(const std::shared_ptr<TensorStorage>& tx_stor,
                 const std::shared_ptr<TensorStorage>& ty_stor,
                 const std::shared_ptr<TensorStorage>& tx_grad,
                 const std::shared_ptr<TensorStorage>& ty_grad,
                 const std::shared_ptr<GradientFlow>& tx_flow,
                 const std::shared_ptr<GradientFlow>& ty_flow
        );

        void backward(MindTensor *_grad) override;
    private:
        std::shared_ptr<MindTensor> tx_;
        std::shared_ptr<MindTensor> ty_;
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRADIENT_OPERATIONS_HPP
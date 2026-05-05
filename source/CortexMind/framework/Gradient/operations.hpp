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
        addition(const std::shared_ptr<TensorStorage>& tx_stor, const std::shared_ptr<TensorStorage>& ty_stor, const std::shared_ptr<TensorStorage>& tx_grad_stor, const std::shared_ptr<TensorStorage>& ty_grad_stor, const std::shared_ptr<GradientFlow>& tx_flow, const std::shared_ptr<GradientFlow>& ty_flow);
        ~addition() override;

        void backward(MindTensor *_grad) override;
    private:
        MindTensor* tx;
        MindTensor* ty;
    };

    struct subtraction : GradientFlow {
        subtraction(const std::shared_ptr<TensorStorage>& tx_stor, const std::shared_ptr<TensorStorage>& ty_stor, const std::shared_ptr<TensorStorage>& tx_grad_stor, const std::shared_ptr<TensorStorage>& ty_grad_stor, const std::shared_ptr<GradientFlow>& tx_flow, const std::shared_ptr<GradientFlow>& ty_flow);
        ~subtraction() override;

        void backward(MindTensor *_grad) override;
    private:
        MindTensor* tx;
        MindTensor* ty;
    };

    struct multiply : GradientFlow {
        multiply(const std::shared_ptr<TensorStorage>& tx_stor, const std::shared_ptr<TensorStorage>& ty_stor, const std::shared_ptr<TensorStorage>& tx_grad_stor, const std::shared_ptr<TensorStorage>& ty_grad_stor, const std::shared_ptr<GradientFlow>& tx_flow, const std::shared_ptr<GradientFlow>& ty_flow);
        ~multiply() override;

        void backward(MindTensor *_grad) override;
    private:
        MindTensor* tx;
        MindTensor* ty;
    };

    struct division : GradientFlow {
        division(const std::shared_ptr<TensorStorage>& tx_stor, const std::shared_ptr<TensorStorage>& ty_stor, const std::shared_ptr<TensorStorage>& tx_grad_stor, const std::shared_ptr<TensorStorage>& ty_grad_stor, const std::shared_ptr<GradientFlow>& tx_flow, const std::shared_ptr<GradientFlow>& ty_flow);
        ~division() override;

        void backward(MindTensor *_grad) override;
    private:
        MindTensor* tx;
        MindTensor* ty;
    };

    struct scalar_additive : GradientFlow {
        scalar_additive(const std::shared_ptr<TensorStorage>& tx_stor, const std::shared_ptr<TensorStorage>& tx_grad_stor, const std::shared_ptr<GradientFlow>& tx_flow);
        ~scalar_additive() override;

        void backward(MindTensor *_grad) override;
    private:
        MindTensor* tx;
    };

    struct scalar_multiply : GradientFlow {
        scalar_multiply(const std::shared_ptr<TensorStorage>& tx_stor, const std::shared_ptr<TensorStorage>& tx_grad_stor, const std::shared_ptr<GradientFlow>& tx_flow, f32 c);
        ~scalar_multiply() override;

        void backward(MindTensor *_grad) override;
    private:
        MindTensor* tx;
        f32 c;
    };

    struct dot : GradientFlow {
        dot(const std::shared_ptr<TensorStorage>& tx_stor, const std::shared_ptr<TensorStorage>& ty_stor, const std::shared_ptr<TensorStorage>& tx_grad_stor, const std::shared_ptr<TensorStorage>& ty_grad_stor, const std::shared_ptr<GradientFlow>& tx_flow, const std::shared_ptr<GradientFlow>& ty_flow);
        ~dot() override;

        void backward(MindTensor *_grad) override;
    private:
        MindTensor* tx;
        MindTensor* ty;
    };

    struct pow : GradientFlow {
        pow(const std::shared_ptr<TensorStorage>& tx_stor, const std::shared_ptr<TensorStorage>& tx_grad_stor, const std::shared_ptr<GradientFlow>& tx_flow, f32 exp);
        ~pow() override;

        void backward(MindTensor *_grad) override;
    private:
        MindTensor* tx;
        f32 exp;
    };

    struct log : GradientFlow {
        log(const std::shared_ptr<TensorStorage>& tx_stor, const std::shared_ptr<TensorStorage>& tx_grad_stor, const std::shared_ptr<GradientFlow>& tx_flow);
        ~log() override;

        void backward(MindTensor *_grad) override;
    private:
        MindTensor* tx;
    };

    struct exp : GradientFlow {
        exp(const std::shared_ptr<TensorStorage>& tx_stor, const std::shared_ptr<TensorStorage>& tx_grad_stor, const std::shared_ptr<GradientFlow>& tx_flow);
        ~exp() override;

        void backward(MindTensor *_grad) override;
    private:
        MindTensor* tx;
    };

    struct sum : GradientFlow {
        sum(const std::shared_ptr<TensorStorage>& tx_stor, const std::shared_ptr<TensorStorage>& tx_grad_stor, const std::shared_ptr<GradientFlow>& tx_flow);
        ~sum() override;

        void backward(MindTensor *_grad) override;
    private:
        MindTensor* tx;
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRADIENT_OPERATIONS_HPP
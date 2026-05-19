//
// Created by muham on 19.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_GRADIENT_OPERATIONS_HPP
#define CORTEXMIND_FRAMEWORK_GRADIENT_OPERATIONS_HPP

#include <CortexMind/framework/Gradient/flow.hpp>

namespace cortex::_fw::meta {
    struct add : GradientFlow {
        add(const GradientPacked& _x, const GradientPacked& _y);
        ~add() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
        Tensor* ty;
    };

    struct sub : GradientFlow {
        sub(const GradientPacked& _x, const GradientPacked& _y);
        ~sub() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
        Tensor* ty;
    };

    struct mul : GradientFlow {
        mul(const GradientPacked& _x, const GradientPacked& _y);
        ~mul() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
        Tensor* ty;
    };

    struct div : GradientFlow {
        div(const GradientPacked& _x, const GradientPacked& _y);
        ~div() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
        Tensor* ty;
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRADIENT_OPERATIONS_HPP
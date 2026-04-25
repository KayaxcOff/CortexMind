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
        addition(const std::weak_ptr<TensorStorage> &tx_stor, const std::weak_ptr<TensorStorage> &ty_stor, const std::weak_ptr<TensorStorage> &tx_grad_stor, const std::weak_ptr<TensorStorage> &ty_grad_stor, const std::weak_ptr<GradientFlow> &tx_flow, const std::weak_ptr<GradientFlow> &ty_flow);

        void backward(MindTensor *_grad) override;
    private:
        std::weak_ptr<TensorStorage> tx_stor;
        std::weak_ptr<TensorStorage> ty_stor;

        std::weak_ptr<TensorStorage> tx_grad_stor;
        std::weak_ptr<TensorStorage> ty_grad_stor;

        std::weak_ptr<GradientFlow> tx_flow;
        std::weak_ptr<GradientFlow> ty_flow;
    };

    struct subtraction : GradientFlow {
        subtraction(const std::weak_ptr<TensorStorage> &tx_stor, const std::weak_ptr<TensorStorage> &ty_stor, const std::weak_ptr<TensorStorage> &tx_grad_stor, const std::weak_ptr<TensorStorage> &ty_grad_stor, const std::weak_ptr<GradientFlow> &tx_flow, const std::weak_ptr<GradientFlow> &ty_flow);

        void backward(MindTensor *_grad) override;
    private:
        std::weak_ptr<TensorStorage> tx_stor;
        std::weak_ptr<TensorStorage> ty_stor;

        std::weak_ptr<TensorStorage> tx_grad_stor;
        std::weak_ptr<TensorStorage> ty_grad_stor;

        std::weak_ptr<GradientFlow> tx_flow;
        std::weak_ptr<GradientFlow> ty_flow;
    };

    struct multiply : GradientFlow {
        multiply(const std::weak_ptr<TensorStorage> &tx_stor, const std::weak_ptr<TensorStorage> &ty_stor, const std::weak_ptr<TensorStorage> &tx_grad_stor, const std::weak_ptr<TensorStorage> &ty_grad_stor, const std::weak_ptr<GradientFlow> &tx_flow, const std::weak_ptr<GradientFlow> &ty_flow);

        void backward(MindTensor *_grad) override;
    private:
        std::weak_ptr<TensorStorage> tx_stor;
        std::weak_ptr<TensorStorage> ty_stor;

        std::weak_ptr<TensorStorage> tx_grad_stor;
        std::weak_ptr<TensorStorage> ty_grad_stor;

        std::weak_ptr<GradientFlow> tx_flow;
        std::weak_ptr<GradientFlow> ty_flow;
    };

    struct division : GradientFlow {
        division(const std::weak_ptr<TensorStorage> &tx_stor, const std::weak_ptr<TensorStorage> &ty_stor, const std::weak_ptr<TensorStorage> &tx_grad_stor, const std::weak_ptr<TensorStorage> &ty_grad_stor, const std::weak_ptr<GradientFlow> &tx_flow, const std::weak_ptr<GradientFlow> &ty_flow);

        void backward(MindTensor *_grad) override;
    private:
        std::weak_ptr<TensorStorage> tx_stor;
        std::weak_ptr<TensorStorage> ty_stor;

        std::weak_ptr<TensorStorage> tx_grad_stor;
        std::weak_ptr<TensorStorage> ty_grad_stor;

        std::weak_ptr<GradientFlow> tx_flow;
        std::weak_ptr<GradientFlow> ty_flow;
    };

    /**
     * @brief Gradient flow for scalar addition and subtraction.
     * z = x +/- c → dz/dx = 1
     */
    struct scalar_additive : GradientFlow {
        scalar_additive(const std::weak_ptr<TensorStorage>& tx_stor,
                        const std::weak_ptr<TensorStorage>& tx_grad_stor,
                        const std::weak_ptr<GradientFlow>&  tx_flow);
        void backward(MindTensor* _grad) override;
    private:
        std::weak_ptr<TensorStorage> tx_stor;
        std::weak_ptr<TensorStorage> tx_grad_stor;
        std::weak_ptr<GradientFlow>  tx_flow;
    };

    /**
     * @brief Gradient flow for scalar multiplication.
     * z = x * c → dz/dx = c
     */
    struct scalar_multiply : GradientFlow {
        scalar_multiply(const std::weak_ptr<TensorStorage>& tx_stor,
                        const std::weak_ptr<TensorStorage>& tx_grad_stor,
                        const std::weak_ptr<GradientFlow>&  tx_flow,
                        f32 c);
        void backward(MindTensor* _grad) override;
    private:
        std::weak_ptr<TensorStorage> tx_stor;
        std::weak_ptr<TensorStorage> tx_grad_stor;
        std::weak_ptr<GradientFlow>  tx_flow;
        f32 c;
    };

    struct dot : GradientFlow {
        dot(const std::weak_ptr<TensorStorage>& tx_stor, const std::weak_ptr<TensorStorage>& ty_stor, const std::weak_ptr<TensorStorage>& tx_grad_stor, const std::weak_ptr<TensorStorage>& ty_grad_stor, const std::weak_ptr<GradientFlow>&  tx_flow, const std::weak_ptr<GradientFlow>&  ty_flow);

        void backward(MindTensor* _grad) override;
    private:
        std::weak_ptr<TensorStorage> tx_stor;
        std::weak_ptr<TensorStorage> ty_stor;
        std::weak_ptr<TensorStorage> tx_grad_stor;
        std::weak_ptr<TensorStorage> ty_grad_stor;
        std::weak_ptr<GradientFlow>  tx_flow;
        std::weak_ptr<GradientFlow>  ty_flow;
    };

    struct pow : GradientFlow {
        pow(const std::weak_ptr<TensorStorage>& tx_stor, const std::weak_ptr<TensorStorage>& tx_grad_stor, const std::weak_ptr<GradientFlow>&  tx_flow, f32 exp);

        void backward(MindTensor* _grad) override;
    private:
        std::weak_ptr<TensorStorage> tx_stor;
        std::weak_ptr<TensorStorage> tx_grad_stor;
        std::weak_ptr<GradientFlow>  tx_flow;
        f32 exp;
    };

    struct log : GradientFlow {
        log(const std::weak_ptr<TensorStorage>& tx_stor, const std::weak_ptr<TensorStorage>& tx_grad_stor, const std::weak_ptr<GradientFlow>&  tx_flow);

        void backward(MindTensor* _grad) override;
    private:
        std::weak_ptr<TensorStorage> tx_stor;
        std::weak_ptr<TensorStorage> tx_grad_stor;
        std::weak_ptr<GradientFlow>  tx_flow;
    };

    struct exp : GradientFlow {
        exp(const std::weak_ptr<TensorStorage>& tx_stor, const std::weak_ptr<TensorStorage>& tx_grad_stor, const std::weak_ptr<GradientFlow>&  tx_flow);
        void backward(MindTensor* _grad) override;
    private:
        std::weak_ptr<TensorStorage> tx_stor;
        std::weak_ptr<TensorStorage> tx_grad_stor;
        std::weak_ptr<GradientFlow>  tx_flow;
    };

    struct sum : GradientFlow {
        sum(const std::weak_ptr<TensorStorage>& tx_stor, const std::weak_ptr<TensorStorage>& tx_grad_stor, const std::weak_ptr<GradientFlow>&  tx_flow);
        void backward(MindTensor* _grad) override;
    private:
        std::weak_ptr<TensorStorage> tx_stor;
        std::weak_ptr<TensorStorage> tx_grad_stor;
        std::weak_ptr<GradientFlow>  tx_flow;
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRADIENT_OPERATIONS_HPP
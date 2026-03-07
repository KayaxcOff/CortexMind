//
// Created by muham on 24.02.2026.
//

#ifndef CORTEXMIND_CORE_GRAPH_FLOW_OPS_HPP
#define CORTEXMIND_CORE_GRAPH_FLOW_OPS_HPP

#include <CortexMind/core/Graph/flow.hpp>
#include <CortexMind/core/Tools/params.hpp>

namespace cortex::_fw::meta {
    struct add : GradientFlow {
        add(MindTensor* x, MindTensor* y);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        MindTensor* ty;
    };

    struct sub : GradientFlow {
        sub(MindTensor* x, MindTensor* y);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        MindTensor* ty;
    };

    struct mul : GradientFlow {
        mul(MindTensor* x, MindTensor* y);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        MindTensor* ty;
    };

    struct div : GradientFlow {
        div(MindTensor* x, MindTensor* y);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        MindTensor* ty;
    };

    struct matmul : GradientFlow {
        matmul(MindTensor* x, MindTensor* y);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        MindTensor* ty;
    };

    struct sum : GradientFlow {
        explicit
        sum(MindTensor* x);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
    };


    struct sqrt : GradientFlow {
        explicit
        sqrt(MindTensor* x);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
    };

    struct pow : GradientFlow {
        pow(MindTensor* x, f32 exp);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        f32 exp;
    };

    struct exp : GradientFlow {
        explicit
        exp(MindTensor* x);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
    };

    struct log : GradientFlow {
        explicit
        log(MindTensor* x);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
    };

    struct abs : GradientFlow {
        explicit
        abs(MindTensor* x);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
    };

    struct expand : GradientFlow {
        expand(MindTensor* x, std::vector<i64> orig);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        std::vector<i64> original_shape;
    };

    struct repeat : GradientFlow {
        repeat(MindTensor* x, i64 t, i64 d);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        i64 times;
        i64 dim;
    };

    struct relu : GradientFlow {
        explicit
        relu(MindTensor* x);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
    };

    struct tanh : GradientFlow {
        explicit
        tanh(MindTensor* x);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
    };

    struct sigmoid : GradientFlow {
        explicit
        sigmoid(MindTensor* x);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
    };

    struct dropout: GradientFlow {
        dropout(MindTensor* x, MindTensor* y);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        MindTensor* ty;
    };

    struct batch_norm : GradientFlow {
        batch_norm(MindTensor* x,
            MindTensor* gamma,
            MindTensor* beta,
            MindTensor* x_hat,
            MindTensor* mean,
            MindTensor* var,
            f32 eps
        );

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        MindTensor* tgamma;
        MindTensor* tbeta;

        MindTensor* tx_hat;
        MindTensor* tmean;
        MindTensor* tvar;

        f32 eps;
    };

    struct leaky_relu : GradientFlow {
        leaky_relu(MindTensor* x, f32 alpha);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        f32 alpha;
    };

    struct bce : GradientFlow {
        bce(MindTensor* x, MindTensor* y, i64 n, f32 eps);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        MindTensor* ty;
        i64 n;
        f32 eps;
    };
} // namespace cortex::_fw::meta

#endif //CORTEXMIND_CORE_GRAPH_FLOW_OPS_HPP
//
// Created by muham on 16.03.2026.
//

#ifndef CORTEXMIND_CORE_GRAPH_OPS_HPP
#define CORTEXMIND_CORE_GRAPH_OPS_HPP

#include <CortexMind/core/Graph/flow.hpp>

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

    struct sqrt : GradientFlow {
        explicit sqrt(MindTensor* x);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
    };

    struct pow : GradientFlow {
        pow(MindTensor* x, f32 alpha);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        f32 alpha;
    };

    struct exp : GradientFlow {
        explicit exp(MindTensor* x);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
    };

    struct log : GradientFlow {
        explicit log(MindTensor* x);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
    };

    struct abs : GradientFlow {
        explicit abs(MindTensor* x);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
    };

    struct sum_all : GradientFlow {
        explicit sum_all(MindTensor* x);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
    };

    struct sum_dim : GradientFlow {
        sum_dim(MindTensor* x, i64 dim, bool keep);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        i64 dim;
        bool keep;
    };

    struct add_scalar : GradientFlow {
        add_scalar(MindTensor* x, f32 scalar);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        f32 scalar;
    };

    struct sub_scalar : GradientFlow {
        sub_scalar(MindTensor* x, f32 scalar);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        f32 scalar;
    };

    struct mul_scalar : GradientFlow {
        mul_scalar(MindTensor* x, f32 scalar);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        f32 scalar;
    };

    struct div_scalar : GradientFlow {
        div_scalar(MindTensor* x, f32 scalar);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        f32 scalar;
    };

    struct relu : GradientFlow {
        explicit relu(MindTensor* x);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor*> inputs() override;
    private:
        MindTensor* tx;
    };

    struct leaky_relu : GradientFlow {
        explicit leaky_relu(MindTensor* x, f32 alpha);

        void backward(MindTensor &_grad) override;
        std::vector<MindTensor *> inputs() override;
    private:
        MindTensor* tx;
        f32 alpha;
    };
} // namespace cortex::_fw::meta

#endif //CORTEXMIND_CORE_GRAPH_OPS_HPP
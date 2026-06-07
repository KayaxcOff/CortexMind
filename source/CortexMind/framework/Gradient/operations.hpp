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

    struct sum : GradientFlow {
        explicit sum(const GradientPacked& _x);
        ~sum() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct matmul : GradientFlow {
        matmul(const GradientPacked& _x, const GradientPacked& _y);
        ~matmul() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
        Tensor* ty;
    };

    struct pow : GradientFlow {
        pow(const GradientPacked& _x, f32 exp);
        ~pow() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
        f32 exp;
    };

    struct sqrt: GradientFlow {
        explicit sqrt(const GradientPacked& _x);
        ~sqrt() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct exp : GradientFlow {
        explicit exp(const GradientPacked& _x);
        ~exp() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct log : GradientFlow {
        explicit log(const GradientPacked& _x);
        ~log() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct rsqrt: GradientFlow {
        explicit rsqrt(const GradientPacked& _x);
        ~rsqrt() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct sin : GradientFlow {
        explicit sin(const GradientPacked& _x);
        ~sin() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct cos : GradientFlow {
        explicit cos(const GradientPacked& _x);
        ~cos() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct abs : GradientFlow {
        explicit abs(const GradientPacked& _x);
        ~abs() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct neg : GradientFlow {
        explicit neg(const GradientPacked& _x);
        ~neg() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct scalar_add : GradientFlow {
        explicit scalar_add(const GradientPacked& _x);
        ~scalar_add() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct scalar_sub : GradientFlow {
        explicit scalar_sub(const GradientPacked& _x);
        ~scalar_sub() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct scalar_mul : GradientFlow {
        explicit scalar_mul(const GradientPacked& _x, f32 value);
        ~scalar_mul() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
        f32 value;
    };

    struct scalar_div : GradientFlow {
        explicit scalar_div(const GradientPacked& _x, f32 value);
        ~scalar_div() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
        f32 value;
    };

    struct reshape : GradientFlow {
        explicit reshape(const GradientPacked& _x, std::initializer_list<i64> shape);
        ~reshape() override;

        void backward(const Tensor &_grad) override;
    private:
        std::initializer_list<i64> shape;
        Tensor* tx;
    };

    struct permute : GradientFlow {
        explicit permute(const GradientPacked& _x, std::initializer_list<i64> dims);
        ~permute() override;

        void backward(const Tensor &_grad) override;
    private:
        std::initializer_list<i64> dims;
        Tensor* tx;
    };

    struct clamp : GradientFlow {
        explicit clamp(const GradientPacked& _x, f32 min, f32 max);
        ~clamp() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
        f32 min, max;
    };

    struct transpose : GradientFlow {
        explicit transpose(const GradientPacked& _x);
        ~transpose() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct slice : GradientFlow {
        explicit slice(const GradientPacked& _x, i64 dim, i64 start, i64 end);
        ~slice() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
        i64 dim, start, end;
    };

    struct mean : GradientFlow {
        explicit mean(const GradientPacked& _x);
        ~mean() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct sum_dim : GradientFlow {
        explicit sum_dim(const GradientPacked& _x, std::initializer_list<i64> dims, bool keep_dim);
        ~sum_dim() override;

        void backward(const Tensor &_grad) override;
    private:
        std::initializer_list<i64> dims;
        Tensor* tx;
        bool keep_dim;
    };

    struct mean_dim : GradientFlow {
        explicit mean_dim(const GradientPacked& _x, std::initializer_list<i64> dims, bool keep_dim);
        ~mean_dim() override;

        void backward(const Tensor &_grad) override;
    private:
        std::initializer_list<i64> dims;
        Tensor* tx;
        bool keep_dim;
    };

    struct variance_dim : GradientFlow {
        explicit variance_dim(const GradientPacked& _x, std::initializer_list<i64> dims, bool keep_dim);

        ~variance_dim() override;
        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
        std::initializer_list<i64> dims;
        bool keep_dim;
    };

    struct stdv_dim : GradientFlow {
        explicit stdv_dim(const GradientPacked& _x, const GradientPacked& output, std::initializer_list<i64> dims, bool keep_dim);
        ~stdv_dim() override;

        void backward(const Tensor &_grad) override;
    private:
        std::initializer_list<i64> dims;
        Tensor* tx;
        Tensor* cached_output;
        bool keep_dim;
    };

    struct norm1_dim : GradientFlow {
        norm1_dim(const GradientPacked& _x, std::initializer_list<i64> dims, bool keep_dim);
        ~norm1_dim() override;

        void backward(const Tensor &_grad) override;
    private:
        std::initializer_list<i64> dims;
        Tensor* tx;
        bool keep_dim;
    };

    struct norm2_dim : GradientFlow {
        norm2_dim(const GradientPacked& _x, const GradientPacked& output, std::initializer_list<i64> dims, bool keep_dim);
        ~norm2_dim() override;

        void backward(const Tensor &_grad) override;
    private:
        std::initializer_list<i64> dims;
        Tensor* tx;
        Tensor* cached_output;
        bool keep_dim;
    };

    struct variance : GradientFlow {
        explicit variance(const GradientPacked& _x);
        ~variance() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct stdv : GradientFlow {
        explicit stdv(const GradientPacked& _x, const GradientPacked& output);
        ~stdv() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
        Tensor* cached_output;
    };

    struct norm1 : GradientFlow {
        explicit norm1(const GradientPacked& _x);
        ~norm1() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct norm2 : GradientFlow {
        explicit norm2(const GradientPacked& _x, const GradientPacked& output);
        ~norm2() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
        Tensor* cached_output;
    };

    struct exp2 : GradientFlow {
        explicit exp2(const GradientPacked& _x);
        ~exp2() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct exp10 : GradientFlow {
        explicit exp10(const GradientPacked& _x);
        ~exp10() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct log2 : GradientFlow {
        explicit log2(const GradientPacked& _x);
        ~log2() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct log10 : GradientFlow {
        explicit log10(const GradientPacked& _x);
        ~log10() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct square : GradientFlow {
        explicit square(const GradientPacked& _x);
        ~square() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct tan : GradientFlow {
        explicit tan(const GradientPacked& _x);
        ~tan() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct cot : GradientFlow {
        explicit cot(const GradientPacked& _x);
        ~cot() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct sign : GradientFlow {
        explicit sign(const GradientPacked& _x);
        ~sign() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct erf : GradientFlow {
        explicit erf(const GradientPacked& _x);
        ~erf() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct inv : GradientFlow {
        explicit inv(const GradientPacked& _x);
        ~inv() override;

        void backward(const Tensor &_grad) override;
    private:
        Tensor* tx;
    };

    struct relu : GradientFlow {
        explicit relu(const GradientPacked& _x);
        ~relu() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
    };

    struct tanh : GradientFlow {
        explicit tanh(const GradientPacked& _x, const GradientPacked& _y);
        ~tanh() override;
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        Tensor* cached_output;
    };


    struct sigmoid : GradientFlow {

        explicit sigmoid(const GradientPacked& _x, const GradientPacked& _y);
        ~sigmoid() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        Tensor* cached_output;
    };

    struct gelu : GradientFlow {
        explicit gelu(const GradientPacked& _x, const GradientPacked& _y);
        ~gelu() override;
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        Tensor* cached_output;
    };

    struct leaky_relu : GradientFlow {
        explicit leaky_relu(const GradientPacked& _x, f32 alpha);
        ~leaky_relu() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        f32 alpha;
    };

    struct gelu_exact : GradientFlow {
        explicit gelu_exact(const GradientPacked& _x, const GradientPacked& _y);
        ~gelu_exact() override;
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        Tensor* cached_output;
    };

    struct silu : GradientFlow {
        explicit silu(const GradientPacked& _x, const GradientPacked& _y);
        ~silu() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        Tensor* cached_output;
    };

    struct conv2d : GradientFlow {
        explicit conv2d(const GradientPacked& _x, const GradientPacked& _weight, const GradientPacked& _bias, i64 iH, i64 iW, i64 kH, i64 kW, i64 sH, i64 sW, i64 pH, i64 pW, i64 oH, i64 oW);
        ~conv2d() override;
        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        Tensor* tw;
        Tensor* tb;
        i64 iH, iW, kH, kW, sH, sW, pH, pW, oH, oW;
    };

    struct softmax : GradientFlow {
        explicit softmax(const GradientPacked& _x, const GradientPacked& _y);
        ~softmax() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        Tensor* cached_output;
    };

    struct logit_loss : GradientFlow {
        explicit logit_loss(const GradientPacked& _x, const GradientPacked& _y);
        ~logit_loss() override;

        void backward(const Tensor& _grad) override;
    private:
        Tensor* tx;
        Tensor* cached_output;
    };
} //namespace cortex::_fw::meta

#endif //CORTEXMIND_FRAMEWORK_GRADIENT_OPERATIONS_HPP
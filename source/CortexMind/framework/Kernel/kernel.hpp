//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_KERNEL_HPP
#define CORTEXMIND_KERNEL_HPP

#include <CortexMind/framework/Params/params.hpp>

namespace cortex::tools {
    class MindKernel {
    public:
        MindKernel(size_t c_in, size_t c_out, size_t kernel_size, size_t _stride = 1, size_t _padding = 0, bool required_grad = true);
        ~MindKernel();

        [[nodiscard]] tensor get_weights() const {return this->weights;}
        [[nodiscard]] tensor apply(const tensor& input);
        [[nodiscard]] tensor backward(const tensor &input, const tensor& grad_output);

        tensor& get_weights() { return this->weights; }
        tensor& get_bias() { return this->bias; }
        tensor& get_grad_weights() { return this->gradWeights; }
        tensor& get_grad_bias() { return this->gradBias; }
    private:
        tensor weights;
        tensor bias;

        tensor gradWeights;
        tensor gradBias;

        size_t stride, padding;
        size_t K, C_in, C_out;

        tensor padded_input;
        size_t H_in{}, W_in{};
    };
}

#endif //CORTEXMIND_KERNEL_HPP
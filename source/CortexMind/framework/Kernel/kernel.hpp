//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_KERNEL_HPP
#define CORTEXMIND_KERNEL_HPP

#include <CortexMind/framework/Params/params.hpp>

namespace cortex::tools {
    class MindKernel {
    public:
        MindKernel(size_t in, size_t out, size_t kernel_size, size_t _stride = 1, size_t _padding = 0, bool required_grad = true);
        ~MindKernel();

        [[nodiscard]] tensor get_weights() const {return this->weights;}
        [[nodiscard]] tensor apply(const tensor& input);
        [[nodiscard]] tensor backward(const tensor &input, const tensor& grad_output);
    private:
        tensor weights;
        tensor grad;

        size_t stride, padding;
    };
}

#endif //CORTEXMIND_KERNEL_HPP
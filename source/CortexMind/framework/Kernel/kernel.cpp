//
// Created by muham on 30.11.2025.
//

#include "CortexMind/framework/Kernel/kernel.hpp"

using namespace cortex::tools;
using namespace cortex;

MindKernel::MindKernel(const size_t in, const size_t out, const size_t kernel_size, const size_t _stride, const size_t _padding, const bool required_grad) : weights(0, 0, 0), stride(_stride), padding(_padding) {
    this->weights = tensor(out, in, kernel_size * kernel_size, required_grad);
    this->weights.uniform_rand(-0.1, 1.0);
}

MindKernel::~MindKernel() = default;

tensor MindKernel::apply(const tensor &input) const {
    const size_t B = input._shape()[0];
    const size_t C_in = input._shape()[1];
    const size_t H_in = input._shape()[2];
    const size_t W_in = input._shape()[3];

    const size_t K = this->weights._shape()[2];
    const size_t C_out = this->weights._shape()[0];

    const size_t H_out = (H_in - K + 2 * this->padding) / this->stride + 1;
    const size_t W_out = (W_in - K + 2 * this->padding) / this->stride + 1;

    tensor output(B, C_out, H_out * W_out, false);

    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < C_out; ++j) {
            double sum = 0.0;
            for (size_t m = 0; m < C_in; ++m) {
                for (size_t n = 0; n < K; ++n) {
                    sum += input(i, m, n) * this->weights(j, m, n);
                }
            }
            output(i, j, 0) = sum;
        }
    }
    return output;
}
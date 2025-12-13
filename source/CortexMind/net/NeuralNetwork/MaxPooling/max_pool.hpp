//
// Created by muham on 12.12.2025.
//

#ifndef CORTEXMIND_MAX_POOL_HPP
#define CORTEXMIND_MAX_POOL_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    class MaxPooling : public _fw::Layer {
    public:
        MaxPooling(int kernel_width, int kernel_height, int stride_width, int stride_height);
        ~MaxPooling() override = default;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;
        std::string config() override;
    private:
        int KERNEL_WIDTH, KERNEL_HEIGHT;
        int STRIDE_WIDTH, STRIDE_HEIGHT;
        tensor idx;
        int INPUT_CHANNELS, INPUT_WIDTH, INPUT_HEIGHT;
    };
}

#endif //CORTEXMIND_MAX_POOL_HPP
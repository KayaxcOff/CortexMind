//
// Created by muham on 7.12.2025.
//

#ifndef CORTEXMIND_MAX_POOL_HPP
#define CORTEXMIND_MAX_POOL_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    class MaxPooling : public _fw::Layer {
    public:
        MaxPooling(int kernel_width, int kernel_height, int stride_width, int stride_height, std::unique_ptr<_fw::ActivationFunc> activation_func);
        ~MaxPooling() override;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;
        [[nodiscard]] std::string config() const override;
        std::vector<std::reference_wrapper<tensor>> gradients() override;
        std::vector<std::reference_wrapper<tensor>> parameters() override;
    private:
        int KERNEL_WIDTH, KERNEL_HEIGHT;
        int STRIDE_WIDTH, STRIDE_HEIGHT;
        tensor idx;
        int INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS;
    };
}

#endif //CORTEXMIND_MAX_POOL_HPP
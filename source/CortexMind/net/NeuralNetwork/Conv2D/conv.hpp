//
// Created by muham on 7.12.2025.
//

#ifndef CORTEXMIND_CONV_HPP
#define CORTEXMIND_CONV_HPP

#include <CortexMind/framework/Net/layer.hpp>
#include <CortexMind/framework/Kernel/kernel.hpp>
#include <memory>

namespace cortex::nn {
    class Conv2D : _fw::Layer {
    public:
        explicit Conv2D(std::unique_ptr<_fw::ActivationFunc> activation_func, int in_channels = 1, int out_channels = 1, int kernel_height = 1, int kernel_width = 1);
        ~Conv2D() override;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;
        [[nodiscard]] std::string config() const override;
        std::vector<std::reference_wrapper<tensor>> gradients() override;
        std::vector<std::reference_wrapper<tensor>> parameters() override;
    private:
        std::unique_ptr<_fw::ConvKernel> conv_kernel_;
        int KERNEL_HEIGHT, KERNEL_WIDTH;
        int STRIDE_HEIGHT = 1, STRIDE_WIDTH = 1;
        int PADDING_HEIGHT = 0, PADDING_WIDTH = 0;
    };
}

#endif //CORTEXMIND_CONV_HPP
//
// Created by muham on 11.12.2025.
//

#ifndef CORTEXMIND_CONV2D_HPP
#define CORTEXMIND_CONV2D_HPP

#include <CortexMind/framework/Net/layer.hpp>
#include <CortexMind/framework/Tools/Kernel/kernel.hpp>
#include <memory>

namespace cortex::nn {
    class Conv2D : public _fw::Layer {
    public:
        Conv2D(int in_channel, int out_channel, int kernel_width, int kernel_height);
        ~Conv2D() override;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;
        std::string config() override;
    private:
        std::unique_ptr<_fw::MindKernel> mind_kernel_;
    };
}

#endif //CORTEXMIND_CONV2D_HPP
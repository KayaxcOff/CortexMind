//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_CONV_HPP
#define CORTEXMIND_CONV_HPP

#include <CortexMind/framework/NetBase/layer.hpp>
#include <CortexMind/framework/Kernel/kernel.hpp>
#include <memory>

namespace cortex::nn {
    class Conv2D : public Layer {
    public:
        Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0);
        ~Conv2D() override = default;

        tensor forward(tensor &input) override;
        tensor backward(tensor &grad_output) override;
        std::vector<tensor*> getGradients() override;
        std::vector<tensor*> getParameters() override;
        std::string config() override;
    private:
        std::unique_ptr<tools::MindKernel> mind_kernel_;
        tensor input_cache;
    };
}

#endif //CORTEXMIND_CONV_HPP
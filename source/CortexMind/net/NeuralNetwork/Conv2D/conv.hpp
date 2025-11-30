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
        Conv2D();
        ~Conv2D() override;

        tensor forward(tensor &input) override;
        tensor backward(tensor &grad_output) override;
        std::vector<tensor*> getGradients() override;
        std::vector<tensor*> getParameters() override;
        std::string config() override;
    private:
        std::unique_ptr<tools::MindKernel> mind_kernel_;
    };
}

#endif //CORTEXMIND_CONV_HPP
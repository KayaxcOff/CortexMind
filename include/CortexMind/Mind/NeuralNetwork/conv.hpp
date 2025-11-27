//
// Created by muham on 8.11.2025.
//

#ifndef CORTEXMIND_CONV_HPP
#define CORTEXMIND_CONV_HPP

#include <CortexMind/Mind/NeuralNetwork/layer.hpp>
#include <CortexMind/Utils/MathTools/kernel.hpp>
#include <CortexMind/Utils/params.hpp>
#include <memory>

namespace cortex::nn {
    class Conv2D : public Layer {
    public:
        Conv2D();
        ~Conv2D() override;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;

    private:
        tensor weights;
        tensor biases;

        tensor lastOutput;
        tensor lastInput;

        std::unique_ptr<tools::MindKernel> mind_kernel_;
    };
}

#endif //CORTEXMIND_CONV_HPP
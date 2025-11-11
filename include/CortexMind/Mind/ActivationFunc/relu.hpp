//
// Created by muham on 10.11.2025.
//

#ifndef CORTEXMIND_RELU_HPP
#define CORTEXMIND_RELU_HPP

#include <CortexMind/Mind/ActivationFunc/activation.hpp>

namespace cortex::act {
    class ReLU final : public Activation {
    public:
        ReLU();
        ~ReLU() override;

        tensor forward(const tensor& input) override;
        tensor backward(const tensor& grad_output) override;
    };
}

#endif //CORTEXMIND_RELU_HPP
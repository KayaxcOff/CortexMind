//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_RELU_HPP
#define CORTEXMIND_RELU_HPP

#include <CortexMind/framework/NetBase/activation.hpp>

namespace cortex::act {
    class ReLU : public Activation {
    public:
        ReLU();
        ~ReLU() override = default;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;
    private:
        tensor cached_relu;
    };
}

#endif //CORTEXMIND_RELU_HPP
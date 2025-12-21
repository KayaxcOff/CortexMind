//
// Created by muham on 15.12.2025.
//

#ifndef CORTEXMIND_RELU_HPP
#define CORTEXMIND_RELU_HPP

#include <CortexMind/framework/Net/activ.hpp>

namespace cortex::net {
    class ReLU : public _fw::Activation {
    public:
        ReLU() = default;
        ~ReLU() override = default;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;
    private:
        tensor output;
    };
}

#endif //CORTEXMIND_RELU_HPP
//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_LEAKY_HPP
#define CORTEXMIND_LEAKY_HPP

#include "CortexMind/framework/NetBase/activation.hpp"

namespace cortex::act {
    class LeakyReLU : public Activation {
    public:
        LeakyReLU() = default;
        ~LeakyReLU() override = default;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &input) override;
    };
}

#endif //CORTEXMIND_LEAKY_HPP
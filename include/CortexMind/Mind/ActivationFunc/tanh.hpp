//
// Created by muham on 28.11.2025.
//

#ifndef CORTEXMIND_TANH_HPP
#define CORTEXMIND_TANH_HPP

#include <CortexMind/Mind/ActivationFunc/activation.hpp>

namespace cortex::act {
    class Tanh : public Activation {
    public:
        Tanh();
        ~Tanh() override;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;
    private:
        tensor cached_input;
    };
}

#endif //CORTEXMIND_TANH_HPP
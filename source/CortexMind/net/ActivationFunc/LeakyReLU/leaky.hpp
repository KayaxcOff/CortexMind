//
// Created by muham on 15.12.2025.
//

#ifndef CORTEXMIND_LEAKY_HPP
#define CORTEXMIND_LEAKY_HPP

#include <CortexMind/framework/Net/activ.hpp>

namespace cortex::net {
    class LeakyReLU : public _fw::Activation {
    public:
        explicit LeakyReLU(const float _alpha = 0.01f) : alpha(_alpha) {}
        ~LeakyReLU() override = default;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;
    private:
        tensor output;
        float alpha;
    };
}

#endif //CORTEXMIND_LEAKY_HPP
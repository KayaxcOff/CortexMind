//
// Created by muham on 15.12.2025.
//

#ifndef CORTEXMIND_SIGMOID_HPP
#define CORTEXMIND_SIGMOID_HPP

#include <CortexMind/framework/Net/activ.hpp>

namespace cortex::net {
    class Sigmoid : public _fw::Activation {
    public:
        Sigmoid();
        ~Sigmoid() override;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &input) override;
    private:
        tensor output;
    };
}

#endif //CORTEXMIND_SIGMOID_HPP
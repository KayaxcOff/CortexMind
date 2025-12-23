//
// Created by muham on 23.12.2025.
//

#ifndef CORTEXMIND_SOFTMAX_HPP
#define CORTEXMIND_SOFTMAX_HPP

#include <CortexMind/framework/Net/activ.hpp>

namespace cortex::net {
    class Softmax : _fw::Activation {
    public:
        Softmax();
        ~Softmax() override;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;
    private:
        tensor output;
    };
}

#endif //CORTEXMIND_SOFTMAX_HPP
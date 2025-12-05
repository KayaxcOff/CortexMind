//
// Created by muham on 5.12.2025.
//

#ifndef CORTEXMIND_SOFTMAX_HPP
#define CORTEXMIND_SOFTMAX_HPP

#include <CortexMind/framework/NetBase/activation.hpp>

namespace cortex::act {
    class Softmax : public Activation {
    public:
        Softmax();
        ~Softmax() override = default;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;
    private:
        tensor outputCache;
    };
}

#endif //CORTEXMIND_SOFTMAX_HPP
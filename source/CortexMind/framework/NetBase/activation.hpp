//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_ACTIVATION_HPP
#define CORTEXMIND_ACTIVATION_HPP

#include <CortexMind/framework/Params/params.hpp>

namespace cortex::act {
    class Activation {
    public:
        Activation() = default;
        virtual ~Activation() = default;

        virtual tensor forward(const tensor& input) = 0;
        virtual tensor backward(const tensor& grad_output) = 0;
    };
}

#endif //CORTEXMIND_ACTIVATION_HPP
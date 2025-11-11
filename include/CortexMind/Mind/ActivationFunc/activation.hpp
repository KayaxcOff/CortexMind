//
// Created by muham on 8.11.2025.
//

#ifndef CORTEXMIND_ACTIVATION_HPP
#define CORTEXMIND_ACTIVATION_HPP

#include <CortexMind/Utils/params.hpp>

namespace cortex::act {
    class Activation {
    public:
        Activation();
        virtual ~Activation();

        virtual tensor forward(const tensor& input);
        virtual tensor backward(const tensor& grad_output);
    };
}

#endif //CORTEXMIND_ACTIVATION_HPP
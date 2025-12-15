//
// Created by muham on 10.12.2025.
//

#ifndef CORTEXMIND_ACTIV_HPP
#define CORTEXMIND_ACTIV_HPP

#include <CortexMind/framework/Params/params.hpp>

namespace cortex::_fw {
    class Activation {
    public:
        Activation() = default;
        virtual ~Activation() = default;

        [[nodiscard]] virtual tensor forward(const tensor& input) = 0;
        [[nodiscard]] virtual tensor backward(const tensor& grad_output) = 0;
    };
}

#endif //CORTEXMIND_ACTIV_HPP
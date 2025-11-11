//
// Created by muham on 8.11.2025.
//

#ifndef CORTEXMIND_OPTIMIZER_HPP
#define CORTEXMIND_OPTIMIZER_HPP

#include <CortexMind/Utils/params.hpp>

namespace cortex::optim {
    class Optimizer {
    public:
        Optimizer();
        virtual ~Optimizer();

        virtual tensor forward(const tensor& input);
        virtual tensor backward(const tensor& grad_output);
    };
}

#endif //CORTEXMIND_OPTIMIZER_HPP
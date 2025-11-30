//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_OPTIMIZER_HPP
#define CORTEXMIND_OPTIMIZER_HPP

#include <CortexMind/framework/Params/params.hpp>

namespace cortex::optim {
    class Optimizer {
    public:
        explicit Optimizer(const double _lr) : learning_rate(_lr) {}
        virtual ~Optimizer() = default;

        virtual void step(tensor& weights, tensor& grads) = 0;
    protected:
        double learning_rate;
    };
}

#endif //CORTEXMIND_OPTIMIZER_HPP
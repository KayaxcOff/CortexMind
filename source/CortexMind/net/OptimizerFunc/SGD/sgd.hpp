//
// Created by muham on 3.12.2025.
//

#ifndef CORTEXMIND_SGD_HPP
#define CORTEXMIND_SGD_HPP

#include <CortexMind/framework/NetBase/optimizer.hpp>

namespace cortex::optim {
    class StochasticGradient : public Optimizer {
    public:
        StochasticGradient(float learning_rate = 0.01f);
        ~StochasticGradient() override = default;

        void step() override;
    };
}

#endif //CORTEXMIND_SGD_HPP
//
// Created by muham on 16.12.2025.
//

#ifndef CORTEXMIND_MOMENTUM_HPP
#define CORTEXMIND_MOMENTUM_HPP

#include <CortexMind/framework/Net/optim.hpp>

namespace cortex::net {
    class Momentum : _fw::Optimizer {
    public:
        explicit Momentum(double lr = 0.01, double momentum = 0.9);
        ~Momentum() override = default;

        void step() override;
    private:
        tensor input_cache;
        double beta;
    };
}

#endif //CORTEXMIND_MOMENTUM_HPP
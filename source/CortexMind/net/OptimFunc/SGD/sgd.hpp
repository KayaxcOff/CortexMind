//
// Created by muham on 16.12.2025.
//

#ifndef CORTEXMIND_SGD_HPP
#define CORTEXMIND_SGD_HPP

#include <CortexMind/framework/Net/optim.hpp>

namespace cortex::net {
    class StaticGradient : public _fw::Optimizer {
    public:
        explicit StaticGradient(const double _lr = 0.001) : Optimizer(_lr) {}

        void step() override;
    };
}

#endif //CORTEXMIND_SGD_HPP
//
// Created by muham on 1.03.2026.
//

#ifndef CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_SGD_HPP
#define CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_SGD_HPP

#include <CortexMind/core/Net/optimization.hpp>

namespace cortex::opt {
    class StochasticGradient : public _fw::Optimization {
    public:
        explicit
        StochasticGradient(float32 _lr);
        ~StochasticGradient() override;

        void update() override;
    };
} // namespace cortex::opt

#endif //CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_SGD_HPP
//
// Created by muham on 18.03.2026.
//

#ifndef CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_SGD_HPP
#define CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_SGD_HPP

#include <CortexMind/core/Net/optimization.hpp>

namespace cortex::opt {
    /**
     * @brief   Vanilla Stochastic Gradient Descent optimizer
     *
     * Performs simple gradient descent updates without momentum or adaptive learning rates.
     */
    class StochasticGradient : public _fw::Optimization {
    public:
        /**
         * @brief   Constructs SGD optimizer with given learning rate
         * @param   _lr     Learning rate (step size)
         *
         * @note    Name is fixed to "SGD"
         * @note    Learning rate can be modified later via .lr member
         */
        explicit StochasticGradient(float32 _lr = 0.001f);
        ~StochasticGradient() override;

        /**
         * @brief   Performs one SGD update step
         *
         * For each parameter:
         *     param -= lr × grad
         *
         * @note    In-place update — modifies parameter tensors directly
         * @note    Assumes gradients are already computed and accumulated
         * @note    No momentum, weight decay, or adaptive scaling
         */
        void update() override;
    };
} // namespace cortex::opt

#endif //CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_SGD_HPP
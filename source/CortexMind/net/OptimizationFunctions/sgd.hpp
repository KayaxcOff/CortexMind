//
// Created by muham on 5.05.2026.
//

#ifndef CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_SGD_HPP
#define CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_SGD_HPP

#include <CortexMind/framework/Net/optimization.hpp>

namespace cortex::opt {
    /**
     * @brief Stochastic Gradient Descent (SGD) optimizer.
     *
     * Classic gradient descent optimizer that updates parameters using:
     * `parameter = parameter - learning_rate * gradient`
     */
    class StochasticGradient : public _fw::OptimizationBase {
    public:
        /**
         * @brief Constructs SGD optimizer.
         *
         * @param _lr Learning rate (default: 0.0001)
         */
        explicit StochasticGradient(float32 _lr = 0.0001f);
        ~StochasticGradient() override;

        /**
         * @brief Performs parameter update for all registered parameters.
         *
         * Updates every parameter using its gradient:
         * `p = p - lr * p.grad()`
         */
        void update() override;
    };
} //namespace cortex::opt

#endif //CORTEXMIND_NET_OPTIMIZATION_FUNCTIONS_SGD_HPP
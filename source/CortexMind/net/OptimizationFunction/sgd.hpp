//
// Created by muham on 20.05.2026.
//

#ifndef CORTEXMIND_NET_OPTIMIZATION_FUNCTION_SGD_HPP
#define CORTEXMIND_NET_OPTIMIZATION_FUNCTION_SGD_HPP

#include <CortexMind/framework/Net/optimization.hpp>

namespace cortex::opt {
    /**
     * @brief Stochastic Gradient Descent (SGD) optimizer.
     *
     * Classic SGD optimizer that updates parameters using the simple rule:
     *
     * `θ = θ - lr * ∇L(θ)`
     *
     * This is the most basic optimizer and serves as a foundation for more
     * advanced optimizers like Momentum, Adam, etc.
     */
    class StochasticGradient : public _fw::OptimizationBase {
    public:
        /**
         * @brief Constructs a Stochastic Gradient Descent optimizer.
         *
         * @param _lr Learning rate (default: 0.0001)
         */
        explicit StochasticGradient(float32 _lr = 0.0001f);
        ~StochasticGradient() override;

        /**
         * @brief Performs a single optimization step.
         *
         * Updates all registered parameters using the formula:
         *
         * `parameter = parameter - learning_rate * parameter.grad()`
         *
         * This method should be called after `backward()` pass.
         */
        void update() override;
    };
} //namespace cortex::opt

#endif //CORTEXMIND_NET_OPTIMIZATION_FUNCTION_SGD_HPP
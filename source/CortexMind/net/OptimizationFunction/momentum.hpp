//
// Created by muham on 25.05.2026.
//

#ifndef CORTEXMIND_NET_OPTIMIZATION_FUNCTION_MOMENTUM_HPP
#define CORTEXMIND_NET_OPTIMIZATION_FUNCTION_MOMENTUM_HPP

#include <CortexMind/framework/Net/optimization.hpp>

namespace cortex::opt {
    /**
     * @brief Momentum optimizer (SGD with momentum).
     *
     * Classic momentum optimizer that accelerates gradient descent by accumulating
     * a velocity vector in the direction of persistent gradients.
     *
     * Update rule:
     *
     *     v = β * v + ∇L(θ)
     *     θ = θ - lr * v
     *
     * This helps accelerate convergence and dampen oscillations in relevant directions.
     */
    class Momentum : public _fw::OptimizationBase {
    public:
        /**
         * @brief Constructs a Momentum optimizer.
         *
         * @param lr    Learning rate
         * @param beta  Momentum coefficient (usually 0.9 or 0.99)
         */
        Momentum(float32 lr, float32 beta);
        ~Momentum() override;

        /**
         * @brief Performs a single optimization step.
         *
         * Updates all registered parameters using momentum.
         */
        void update() override;
    private:
        float32 beta;
        std::vector<tensor> velocities;
        bool initialized;

        /**
         * @brief Initializes velocity buffers for all parameters.
         *
         * Called automatically on the first update if not already initialized.
         */
        void Init();
    };
} //namespace cortex::opt

#endif //CORTEXMIND_NET_OPTIMIZATION_FUNCTION_MOMENTUM_HPP
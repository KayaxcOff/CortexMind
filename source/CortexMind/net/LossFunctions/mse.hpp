//
// Created by muham on 5.05.2026.
//

#ifndef CORTEXMIND_NET_LOSS_FUNCTIONS_MSE_HPP
#define CORTEXMIND_NET_LOSS_FUNCTIONS_MSE_HPP

#include <CortexMind/framework/Net/loss.hpp>

namespace cortex::loss {
    /**
     * @brief Mean Squared Error (MSE) loss function.
     *
     * Computes the average of squared differences between predictions and targets.
     * Commonly used for regression tasks.
     *
     * Formula:
     * `MSE = (1/N) * Σ (prediction - target)²`
     */
    class MeanSquared : public _fw::LossBase {
    public:
        /**
         * @brief Constructs MeanSquared loss object.
         */
        MeanSquared();
        ~MeanSquared() override;

        /**
         * @brief Computes Mean Squared Error between prediction and target.
         *
         * @param prediction Predicted tensor
         * @param target Ground truth (target) tensor
         * @return Scalar tensor containing the MSE loss value
         */
        tensor forward(tensor &prediction, tensor &target) override;
    };
} //namespace cortex::loss

#endif //CORTEXMIND_NET_LOSS_FUNCTIONS_MSE_HPP
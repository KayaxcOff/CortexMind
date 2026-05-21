//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_NET_LOSS_FUNCTIONS_MAE_HPP
#define CORTEXMIND_NET_LOSS_FUNCTIONS_MAE_HPP

#include <CortexMind/framework/Net/loss.hpp>

namespace cortex::loss {
    /**
     * @brief Mean Absolute Error (MAE) loss function.
     *
     * Computes the average of absolute differences between predictions and targets.
     *
     * Formula:
     *
     *     MAE = (1 / N) * Σ |predict_i - target_i|
     *
     * This loss is more robust to outliers compared to Mean Squared Error.
     */
    class MeanAbsolute : public _fw::LossBase {
    public:
        /**
         * @brief Constructs a Mean Absolute Error loss object.
         */
        MeanAbsolute();
        ~MeanAbsolute() override;

        /**
         * @brief Computes Mean Absolute Error between prediction and target.
         *
         * @param predict Predicted values
         * @param target  Ground truth / target values
         * @return Scalar tensor containing the MAE loss
         */
        [[nodiscard]]
        tensor forward(const tensor &predict, const tensor &target) override;
    };
} //namespace cortex::loss

#endif //CORTEXMIND_NET_LOSS_FUNCTIONS_MAE_HPP
//
// Created by muham on 20.05.2026.
//

#ifndef CORTEXMIND_NET_LOSS_FUNCTION_MSE_HPP
#define CORTEXMIND_NET_LOSS_FUNCTION_MSE_HPP

#include <CortexMind/framework/Net/loss.hpp>

namespace cortex::loss {
    /**
     * @brief Mean Squared Error loss function
     *
     * L = mean((pred - target)²)
     */
    class MeanSquared : public _fw::LossBase {
    public:
        MeanSquared();
        ~MeanSquared() override;

        /**
         * @brief Computes MSE loss
         *
         * @param predict Prediction tensor (batch_size, features)
         * @param target Target tensor (batch_size, features)
         * @return Scalar loss tensor
         */
        [[nodiscard]]
        tensor forward(const tensor &predict, const tensor &target) override;
    };
} //namespace cortex::loss

#endif //CORTEXMIND_NET_LOSS_FUNCTION_MSE_HPP
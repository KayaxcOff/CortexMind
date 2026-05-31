//
// Created by muham on 31.05.2026.
//

#ifndef CORTEXMIND_NET_METRICS_RMSE_HPP
#define CORTEXMIND_NET_METRICS_RMSE_HPP

#include <CortexMind/framework/Net/metric.hpp>

namespace cortex::metric {
    /**
     * @brief Root Mean Squared Error (RMSE) metric.
     *
     * Computes the square root of the average of squared differences between
     * predictions and targets. It is in the same unit as the target values,
     * making it more interpretable than MSE.
     *
     * Formula:
     *
     *     RMSE = √( (1 / N) * Σ (predict_i - target_i)² )
     */
    class RootMeanSquared : public _fw::MetricBase {
    public:
        /**
         * @brief Constructs a Root Mean Squared Error metric.
         */
        RootMeanSquared();
        ~RootMeanSquared() override;

        /**
         * @brief Computes Root Mean Squared Error.
         *
         * @param predict Predicted values
         * @param target  Ground truth values
         * @return RMSE value as a scalar float32
         */
        [[nodiscard]]
        float32 forward(const tensor &predict, const tensor &target) override;
    };
} //namespace cortex::metric

#endif //CORTEXMIND_NET_METRICS_RMSE_HPP
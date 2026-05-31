//
// Created by muham on 31.05.2026.
//

#ifndef CORTEXMIND_NET_METRICS_MSE_HPP
#define CORTEXMIND_NET_METRICS_MSE_HPP

#include <CortexMind/framework/Net/metric.hpp>

namespace cortex::metric {
    /**
     * @brief Mean Squared Error (MSE / L2 Loss) metric.
     *
     * Computes the average of the squared differences between predicted values
     * and target values. Widely used for regression tasks.
     *
     * Formula:
     *
     *     MSE = (1 / N) * Σ (predict_i - target_i)²
     */
    class MeanSquared : public _fw::MetricBase {
    public:
        /**
         * @brief Constructs a Mean Squared Error metric.
         */
        MeanSquared();
        ~MeanSquared() override;

        /**
         * @brief Computes Mean Squared Error between predictions and targets.
         *
         * @param predict Predicted values
         * @param target  Ground truth values
         * @return MSE value as a scalar float32
         */
        [[nodiscard]]
        float32 forward(const tensor &predict, const tensor &target) override;
    };
} //namespace cortex::metric

#endif //CORTEXMIND_NET_METRICS_MSE_HPP
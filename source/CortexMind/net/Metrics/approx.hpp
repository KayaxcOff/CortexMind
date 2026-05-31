//
// Created by muham on 31.05.2026.
//

#ifndef CORTEXMIND_NET_APPROX_ACC_HPP
#define CORTEXMIND_NET_APPROX_ACC_HPP

#include <CortexMind/framework/Net/metric.hpp>

namespace cortex::metric {
    /**
     * @brief Approximate Match (ApproxMatch) metric.
     *
     * A custom evaluation metric that measures the percentage of predictions
     * that are within a small tolerance (default: ±0.1) of the target values.
     *
     * Useful for regression tasks where exact equality is not required,
     * or when evaluating models with continuous outputs that should be close to targets.
     */
    class ApproxMatch : public _fw::MetricBase {
    public:
        /**
         * @brief Constructs an Approximate Match metric.
         */
        ApproxMatch();
        ~ApproxMatch() override;

        /**
         * @brief Computes the approximate match ratio.
         *
         * For each prediction, checks if `|predict - target| <= 0.1`.
         * Returns the ratio of predictions that satisfy this condition.
         *
         * @param predict Predicted values
         * @param target  Ground truth values
         * @return Ratio of approximately correct predictions (in range [0.0, 1.0])
         */
        [[nodiscard]]
        float32 forward(const tensor &predict, const tensor &target) override;
    };
} //namespace cortex::metric

#endif //CORTEXMIND_NET_APPROX_ACC_HPP
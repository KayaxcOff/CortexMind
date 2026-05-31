//
// Created by muham on 31.05.2026.
//

#ifndef CORTEXMIND_NET_METRICS_ACC_HPP
#define CORTEXMIND_NET_METRICS_ACC_HPP

#include <CortexMind/framework/Net/metric.hpp>

namespace cortex::metric {
    /**
     * @brief Classification Accuracy metric.
     *
     * Computes the percentage of correct predictions for binary classification tasks.
     *
     * A prediction is considered correct if:
     * - `predict > 0.5` and `target > 0.5`, or
     * - `predict ≤ 0.5` and `target ≤ 0.5`
     *
     * This metric is typically used after a sigmoid activation in the final layer.
     */
    class Accuracy : public _fw::MetricBase {
    public:
        /**
         * @brief Constructs an Accuracy metric.
         */
        Accuracy();
        ~Accuracy() override;

        /**
         * @brief Computes binary classification accuracy.
         *
         * @param predict Predicted probabilities (after sigmoid)
         * @param target  Ground truth binary labels (0 or 1)
         * @return Accuracy as a value in range [0.0, 1.0]
         */
        [[nodiscard]]
        float32 forward(const tensor &predict, const tensor &target) override;
    };
} //namespace cortex::metric

#endif //CORTEXMIND_NET_METRICS_ACC_HPP
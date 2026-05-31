//
// Created by muham on 31.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_NET_METRIC_HPP
#define CORTEXMIND_FRAMEWORK_NET_METRIC_HPP

#include <CortexMind/tools/types.hpp>
#include <string>

namespace cortex::_fw {
    /**
     * @brief Abstract base class for evaluation metrics.
     *
     * Metrics are used to measure the performance of a model during training
     * and evaluation (e.g. Accuracy, Precision, Recall, F1-Score, etc.).
     *
     * Unlike loss functions, metrics are not used for optimization — they are
     * only for monitoring and reporting.
     */
    class MetricBase {
    public:
        /**
         * @brief Constructs a MetricBase with a given name.
         *
         * @param name Human-readable name of the metric (e.g. "Accuracy", "F1Score")
         */
        explicit MetricBase(std::string name);
        virtual ~MetricBase();

        /**
         * @brief Computes the metric value between predictions and targets.
         *
         * @param predict Predicted values (model output)
         * @param target  Ground truth / target values
         * @return Computed metric value (usually a scalar between 0 and 1)
         */
        [[nodiscard]]
        virtual float32 forward(const tensor& predict, const tensor& target) = 0;
        /**
         * @brief Returns the name of the metric.
         *
         * Used for logging, reporting, and model summary.
         */
        [[nodiscard]]
        const std::string& name() const;
    private:
        std::string m_name;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_NET_METRIC_HPP
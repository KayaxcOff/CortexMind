//
// Created by muham on 5.03.2026.
//

#ifndef CORTEXMIND_UTILS_METRIC_METRIC_HPP
#define CORTEXMIND_UTILS_METRIC_METRIC_HPP

#include <CortexMind/tools/params.hpp>

namespace cortex::utils {
    class Metrics {
    public:
        [[nodiscard]]
        static float32 accuracy(const tensor& predicted, const tensor& target, float32 threshold = 0.5f);
        [[nodiscard]]
        static float32 precision(const tensor& predicted, const tensor& target, float32 threshold = 0.5f);
        [[nodiscard]]
        static float32 recall(const tensor& predicted, const tensor& target, float32 threshold = 0.5f);
        [[nodiscard]]
        static float32 f1(const tensor& predicted, const tensor& target, float32 threshold = 0.5f);

        [[nodiscard]]
        static float32 mse(const tensor& predicted, const tensor& target);
        [[nodiscard]]
        static float32 mae(const tensor& predicted, const tensor& target);
        [[nodiscard]]
        static float32 rmse(const tensor& predicted, const tensor& target);

        static void classification_report(const tensor& predicted, const tensor& target, float32 threshold = 0.5f);
    };
} // namespace cortex::utils

#endif //CORTEXMIND_UTILS_METRIC_METRIC_HPP
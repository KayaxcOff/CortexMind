//
// Created by muham on 31.05.2026.
//

#include "CortexMind/net/Metrics/acc.hpp"

using namespace cortex::_fw;
using namespace cortex::metric;
using namespace cortex;

Accuracy::Accuracy() : MetricBase("Accuracy") {}

Accuracy::~Accuracy() = default;

float32 Accuracy::forward(const tensor &predict, const tensor &target) {
    const size_t batch = predict.shape()[0];

    const float32* p = predict.get();
    const float32* t = target.get();

    size_t correct = 0;

    for (size_t i = 0; i < batch; ++i) {
        const float32 pred = p[i];
        const float32 true_val = t[i];

        const bool pred_label = pred > 0.5f;

        if (const bool true_label = true_val > 0.5f; pred_label == true_label) {
            ++correct;
        }
    }

    return static_cast<float32>(correct) / static_cast<float32>(batch);
}
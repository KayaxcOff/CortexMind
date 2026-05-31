//
// Created by muham on 31.05.2026.
//

#include "CortexMind/net/Metrics/approx.hpp"

using namespace cortex::_fw;
using namespace cortex::metric;
using namespace cortex;

ApproxMatch::ApproxMatch() : MetricBase("ApproxMatch") {}

ApproxMatch::~ApproxMatch() = default;

float32 ApproxMatch::forward(const tensor &predict, const tensor &target) {
    size_t idx = 0;
    const size_t total = predict.len();

    for (size_t i = 0; i < total; ++i) {
        if (const float32 diff = std::abs(predict.get()[i] - target.get()[i]); diff <= 0.1f) {
            idx++;
        }
    }

    return static_cast<float32>(idx) / static_cast<float32>(total);
}
//
// Created by muham on 5.03.2026.
//

#include "CortexMind/utils/Metric/metric.hpp"
#include <CortexMind/core/Tools/err.hpp>
#include <iomanip>
#include <cmath>

using namespace cortex::utils;
using namespace cortex;

static void compute_confusion(const tensor& predicted, const tensor& target, const float32 threshold, int64& tp, int64& fp, int64& tn, int64& fn) {
    CXM_ASSERT(predicted.shape()[0] == target.shape()[0], "cortex::utils::Metrics", "Batch size mismatch.");

    tp = fp = tn = fn = 0;
    const int64 n = predicted.shape()[0];

    for (int64 i = 0; i < n; ++i) {
        const float32 pred  = predicted.at(i, 0) >= threshold ? 1.0f : 0.0f;
        const float32 truth = target.at(i, 0);

        if      (pred == 1.0f && truth == 1.0f) ++tp;
        else if (pred == 1.0f && truth == 0.0f) ++fp;
        else if (pred == 0.0f && truth == 0.0f) ++tn;
        else if (pred == 0.0f && truth == 1.0f) ++fn;
    }
}

float32 Metrics::accuracy(const tensor &predicted, const tensor &target, const float32 threshold) {
    int64 tp, fp, tn, fn;
    compute_confusion(predicted, target, threshold, tp, fp, tn, fn);
    const int64 total = tp + fp + tn + fn;
    return total == 0 ? 0.0f : static_cast<float32>(tp + tn) / static_cast<float32>(total) * 100.0f;
}

float32 Metrics::precision(const tensor &predicted, const tensor &target, const float32 threshold) {
    int64 tp, fp, tn, fn;
    compute_confusion(predicted, target, threshold, tp, fp, tn, fn);
    return (tp + fp) == 0 ? 0.0f : static_cast<float32>(tp) / static_cast<float32>(tp + fp) * 100.0f;
}

float32 Metrics::recall(const tensor &predicted, const tensor &target, const float32 threshold) {
    int64 tp, fp, tn, fn;
    compute_confusion(predicted, target, threshold, tp, fp, tn, fn);
    return (tp + fn) == 0 ? 0.0f : static_cast<float32>(tp) / static_cast<float32>(tp + fn) * 100.0f;
}

float32 Metrics::f1(const tensor &predicted, const tensor &target, const float32 threshold) {
    const float32 p = precision(predicted, target, threshold);
    const float32 r = recall(predicted, target, threshold);
    return (p + r) == 0.0f ? 0.0f : 2.0f * p * r / (p + r);
}

float32 Metrics::mse(const tensor &predicted, const tensor &target) {
    CXM_ASSERT(predicted.shape() == target.shape(), "cortex::utils::Metrics::mse()", "Shape mismatch.");
    const auto n = static_cast<int64>(predicted.numel());
    float32 sum = 0.0f;
    for (int64 i = 0; i < n; ++i) {
        const float32 d = predicted.get()[i] - target.get()[i];
        sum += d * d;
    }
    return sum / static_cast<float32>(n);
}

float32 Metrics::mae(const tensor &predicted, const tensor &target) {
    CXM_ASSERT(predicted.shape() == target.shape(),
               "cortex::utils::Metrics::mae()", "Shape mismatch.");
    const auto n = static_cast<int64>(predicted.numel());
    float32 sum = 0.0f;
    for (int64 i = 0; i < n; ++i) sum += std::abs(predicted.get()[i] - target.get()[i]);
    return sum / static_cast<float32>(n);
}

float32 Metrics::rmse(const tensor &predicted, const tensor &target) {
    return std::sqrt(mse(predicted, target));
}

void Metrics::classification_report(const tensor& predicted, const tensor& target,
                                     const float32 threshold) {
    int64 tp, fp, tn, fn;
    compute_confusion(predicted, target, threshold, tp, fp, tn, fn);

    const float32 acc  = accuracy(predicted,  target, threshold);
    const float32 prec = precision(predicted, target, threshold);
    const float32 rec  = recall(predicted,    target, threshold);
    const float32 f1s  = f1(predicted,        target, threshold);

    const int64 n = predicted.shape()[0];

    printf("╔══════════════════════════════════════╗\n");
    printf("║       Classification Report          ║\n");
    printf("╠══════════════════════════════════════╣\n");
    printf("║ Samples   : %24lld ║\n", n);
    printf("║ Threshold : %23.2f  ║\n", threshold);
    printf("╠══════════════════════════════════════╣\n");
    printf("║ TP:%2lld  FP:%2lld  TN:%2lld  FN:%2lld           ║\n", tp, fp, tn, fn);
    printf("╠══════════════════════════════════════╣\n");
    printf("║ Accuracy  : %20.2f%%    ║\n", acc);
    printf("║ Precision : %20.2f%%    ║\n", prec);
    printf("║ Recall    : %20.2f%%    ║\n", rec);
    printf("║ F1 Score  : %20.2f%%    ║\n", f1s);
    printf("╚══════════════════════════════════════╝\n");
}
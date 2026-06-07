//
// Created by muham on 3.06.2026.
//

#include "CortexMind/net/LossFunction/cce_with_logit.hpp"
#include <CortexMind/framework/Gradient/operations.hpp>
#include <cmath>

using namespace cortex::_fw;
using namespace cortex::loss;
using namespace cortex;

CategoricalCrossEntropyWithLogit::CategoricalCrossEntropyWithLogit() : LossBase("CCEWithLogit") {}

CategoricalCrossEntropyWithLogit::~CategoricalCrossEntropyWithLogit() = default;

tensor CategoricalCrossEntropyWithLogit::forward(const tensor &predict, const tensor &target) {
    const std::span<const i64> shape = predict.shape();
    const auto batch_size = static_cast<size_t>(shape[0]);
    const auto class_count = static_cast<size_t>(shape[1]);

    float32 total_loss = 0.0f;
    const float32* pred_ptr = predict.get();
    const float32* target_ptr = target.get();

    for (size_t b = 0; b < batch_size; ++b) {
        size_t row_offset = b * class_count;
        float32 max_logit = pred_ptr[row_offset];
        for (size_t c = 1; c < class_count; ++c) {
            if (pred_ptr[row_offset + c] > max_logit) max_logit = pred_ptr[row_offset + c];
        }

        float32 sum_exp = 0.0f;
        for (size_t c = 0; c < class_count; ++c) {
            sum_exp += std::exp(pred_ptr[row_offset + c] - max_logit);
        }
        float32 log_sum_exp = std::log(std::max(sum_exp, 1e-12f));

        for (size_t c = 0; c < class_count; ++c) {
            float32 log_prob = (pred_ptr[row_offset + c] - max_logit) - log_sum_exp;
            total_loss -= target_ptr[row_offset + c] * log_prob;
        }
    }

    // Skalar nihai çıkış tensörü oluşturuluyor (requires_grad'ı predict'e bağlıyoruz)
    tensor output({1}, predict.device(), predict.has_grad());
    output.get()[0] = total_loss / static_cast<float32>(batch_size);

    // KOPAN GRAFİĞİ BURADA TAMİR EDİYORUZ:
    if (predict.has_grad()) {
        auto flow = std::make_shared<meta::logit_loss>(predict.pack(), target.pack());
        output.SetFlow(flow);
    }

    return output;
}

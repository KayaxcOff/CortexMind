//
// Created by muham on 6.03.2026.
//

#include "CortexMind/net/LossFunctions/bce.hpp"
#include <CortexMind/core/Tools/err.hpp>

using namespace cortex::loss;
using namespace cortex::_fw;
using namespace cortex;

BinaryCrossEntropy::BinaryCrossEntropy() : Loss("BCE") {}

BinaryCrossEntropy::~BinaryCrossEntropy() = default;

tensor BinaryCrossEntropy::forward(const tensor &predicted, const tensor &target) {
    CXM_ASSERT(predicted.shape() == target.shape(), "cortex::net::BinaryCrossEntropy::forward()", "Shape mismatch.");

    constexpr float32 eps = 1e-7f;

    tensor pred_clamp  = predicted + eps;
    tensor inv_pred    = (predicted * -1.0f) + (1.0f + eps);

    tensor term1 = target          * pred_clamp.log();
    tensor term2 = (target * -1.0f + 1.0f) * inv_pred.log();

    tensor loss_per = (term1 + term2) * -1.0f;
    return loss_per.sum();
}

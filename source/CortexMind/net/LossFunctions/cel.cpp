//
// Created by muham on 8.03.2026.
//

#include "CortexMind/net/LossFunctions/cel.hpp"
#include <CortexMind/core/Tools/err.hpp>

using namespace cortex::loss;
using namespace cortex::_fw;
using namespace cortex;

CrossBinary::CrossBinary() : Loss("CEL") {}

CrossBinary::~CrossBinary() = default;

tensor CrossBinary::forward(const tensor &predicted, const tensor &target) {
    CXM_ASSERT(predicted.shape() == target.shape(), "cortex::loss::CrossBinary::forward()", "Shape mismatch");

    constexpr float32 eps = 1e-7f;

    const tensor log_pred = (predicted + eps).log();
    const tensor loss_per = target * log_pred;

    return loss_per.sum();
}

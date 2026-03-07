//
// Created by muham on 1.03.2026.
//

#include "CortexMind/net/LossFunctions/mae.hpp"
#include <CortexMind/core/Tools/err.hpp>

using namespace cortex::loss;
using namespace cortex::_fw;
using namespace cortex;

MeanAbsolute::MeanAbsolute() : Loss("MAE") {}

MeanAbsolute::~MeanAbsolute() = default;

tensor MeanAbsolute::forward(const tensor &predicted, const tensor &target) {
    CXM_ASSERT(target.shape() == predicted.shape(), "cortex::net::MeanAbsolute::forward()", "Shapes mismatch");

    const tensor diff = target - predicted;
    return diff.abs().sum();
}

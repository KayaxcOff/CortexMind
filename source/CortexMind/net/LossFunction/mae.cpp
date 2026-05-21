//
// Created by muham on 21.05.2026.
//

#include "CortexMind/net/LossFunction/mae.hpp"
#include <CortexMind/framework/Tools/err.hpp>

using namespace cortex::_fw;
using namespace cortex::loss;
using namespace cortex;

MeanAbsolute::MeanAbsolute() : LossBase("MAE") {}

MeanAbsolute::~MeanAbsolute() = default;

tensor MeanAbsolute::forward(const tensor &predict, const tensor &target) {
    CXM_ASSERT(predict.len() != target.len(), "Number elements of predict and target must be same");
    return (target - predict).abs().sum() / static_cast<float32>(predict.len());
}
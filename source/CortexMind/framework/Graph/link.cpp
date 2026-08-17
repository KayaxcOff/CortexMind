//
// Created by muham on 16.08.2026.
//

#include "CortexMind/framework/Graph/link.hpp"
#include <CortexMind/framework/Tensor/tensor.hpp>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

GradientLink::GradientLink(Tensor &t, const std::shared_ptr<GradientFlow> &flow) {
    t.flow_ = flow;
}

GradientLink::~GradientLink() = default;
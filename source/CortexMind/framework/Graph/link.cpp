//
// Created by muham on 16.08.2026.
//

#include "CortexMind/framework/Graph/link.hpp"
#include <CortexMind/framework/Tensor/tensor.hpp>
#include <tlx/utility.hpp>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

GradientLink::GradientLink(Tensor &t, const std::shared_ptr<GradientFlow> &flow) {
    t.flow_ = tlx::move(flow);
}

GradientLink::~GradientLink() = default;
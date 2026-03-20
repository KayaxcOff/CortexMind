//
// Created by muham on 15.03.2026.
//

#include "CortexMind/core/Graph/flow.hpp"

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

GradientFlow::GradientFlow() : idx(global_counter++) {}

GradientFlow::~GradientFlow() = default;

i32 GradientFlow::count() const {
    return this->idx;
}
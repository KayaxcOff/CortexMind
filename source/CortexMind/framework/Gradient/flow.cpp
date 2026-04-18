//
// Created by muham on 18.04.2026.
//

#include "CortexMind/framework/Gradient/flow.hpp"

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

GradientFlow::GradientFlow(const i32 id) : id(id) {}

GradientFlow::~GradientFlow() = default;

size_t GradientFlow::count() const {
    return this->id;
}
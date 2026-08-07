//
// Created by muham on 7.08.2026.
//

#include "CortexMind/framework/Graph/flow.hpp"

using namespace cortex::_fw::meta;

GradientFlow::GradientFlow(const tlx::vstring name) {
    this->m_name = name;
}

GradientFlow::~GradientFlow() = default;

std::string_view GradientFlow::ToString() const noexcept {
    return this->m_name;
}
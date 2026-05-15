//
// Created by muham on 15.05.2026.
//

#include "CortexMind/framework/Gradient/flow.hpp"
#include <type_traits>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

GradientFlow::GradientFlow(std::string _op_name, const i32 ID) : ID(ID), m_name(std::move(_op_name)) {}

GradientFlow::~GradientFlow() = default;

i32 GradientFlow::count() const {
    return this->ID;
}

const std::string &GradientFlow::name() const {
    return this->m_name;
}
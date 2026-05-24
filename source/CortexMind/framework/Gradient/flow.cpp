//
// Created by muham on 15.05.2026.
//

#include "CortexMind/framework/Gradient/flow.hpp"
#include <iostream>
#include <type_traits>

using namespace cortex::_fw::meta;
using namespace cortex::_fw;

GradientFlow::GradientFlow(std::string _op_name) : m_name(std::move(_op_name)) {}

GradientFlow::~GradientFlow() = default;

const std::string &GradientFlow::name() const {
    return this->m_name;
}

void GradientFlow::print() const {
    std::cout << this->m_name << std::endl;
}
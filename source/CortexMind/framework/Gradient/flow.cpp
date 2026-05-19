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

void GradientFlow::save(const std::weak_ptr<GradientFlow> &_flow) {
    this->next_functions.push_back(_flow);
}

void GradientFlow::propagate_backward(const Tensor &_grad) const {
    for (const auto& next_fn_weak : this->next_functions) {
        if (const auto next_fn = next_fn_weak.lock()) {
            next_fn->backward(_grad);
        }
    }
}
//
// Created by muham on 21.04.2026.
//

#include "CortexMind/framework/Net/optimization.hpp"
#include <type_traits>

using namespace cortex::_fw;

OptimizationBase::OptimizationBase(std::string name, const float32 _lr) : m_name(std::move(name)), learning_rate(_lr) {}

OptimizationBase::~OptimizationBase() = default;

void OptimizationBase::setParams(std::vector<ref<tensor>>& _params) {
    this->params = std::move(_params);
}

void OptimizationBase::zero_grad() const {
    for (auto item : this->params) {
        item.get().grad().zero();
    }
}

const std::string& OptimizationBase::name() const {
    return this->m_name;
}
//
// Created by muham on 21.04.2026.
//

#include "CortexMind/framework/Net/optimization.hpp"
#include <type_traits>

using namespace cortex::_fw;

OptimizationBase::OptimizationBase(std::string name, const float32 _lr) : kName(std::move(name)), learning_rate(_lr) {}

OptimizationBase::~OptimizationBase() = default;

void OptimizationBase::setParams(std::vector<ref<tensor>>& params) {
    this->gradients = std::move(params);
}

void OptimizationBase::zero_grad() const {
    for (auto item : this->gradients) {
        item.get().grad().zero();
    }
}

const std::string& OptimizationBase::name() const {
    return this->kName;
}
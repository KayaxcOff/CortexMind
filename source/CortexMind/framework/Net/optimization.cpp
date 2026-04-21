//
// Created by muham on 21.04.2026.
//

#include "CortexMind/framework/Net/optimization.hpp"

using namespace cortex::_fw;
using namespace cortex;

OptimizationBase::OptimizationBase(std::string name, const float32 _lr) : kName(std::move(name)), kLearningRate(_lr) {}

OptimizationBase::~OptimizationBase() = default;

void OptimizationBase::setParams(std::vector<ref<tensor>> params) {
    this->kGradients = std::move(params);
}

void OptimizationBase::zero_grad() const {
    for (auto item : this->kGradients) {
        item.get().grad().zero();
    }
}

const std::string& OptimizationBase::name() const {
    return this->kName;
}
//
// Created by muham on 19.05.2026.
//

#include "CortexMind/framework/Net/optimization.hpp"
#include <CortexMind/framework/Tools/tensor_debug.hpp>

using namespace cortex::_fw;
using namespace cortex;

OptimizationBase::OptimizationBase(std::string name, const float32 _lr) : m_lr(_lr), m_name(std::move(name)) {}

OptimizationBase::~OptimizationBase() = default;

void OptimizationBase::SetLearningRate(const float32 lr) {
    this->m_lr = lr;
}

void OptimizationBase::SetParams(const std::vector<ref<tensor> > &params) {
    this->m_params = params;
}

void OptimizationBase::zero_grad() const {
    if (!this->m_params.empty()) [[likely]] {
        for (auto item : this->m_params) {
            item.get().grad().zero();
        }
    }
}

const std::vector<ref<tensor>> &OptimizationBase::getParameters() {
    CXM_ASSERT(this->m_params.empty() == true, "Parameters isn't initialized");
    return this->m_params;
}

float32 OptimizationBase::lr() const {
    return this->m_lr;
}

const std::string &OptimizationBase::name() {
    return this->m_name;
}
//
// Created by muham on 19.05.2026.
//

#include "CortexMind/framework/Net/optimization.hpp"
#include <CortexMind/framework/Tools/tensor_debug.hpp>

using namespace cortex::_fw;
using namespace cortex;

OptimizationBase::OptimizationBase(std::string name, const float32 _lr) : m_lr(_lr), m_name(std::move(name)), is_initialized(false) {}

OptimizationBase::~OptimizationBase() = default;

void OptimizationBase::SetParams(const std::vector<ref<tensor> > &params) {
    this->m_params = params;
    this->is_initialized = true;
}

void OptimizationBase::SetLearningRate(const float32 lr) {
    this->m_lr = lr;
}

void OptimizationBase::zero_grad() const {
    if (this->is_initialized) [[likely]] {
        for (auto item : this->m_params) {
            item.get().grad().zero();
        }
    }
}

const std::vector<ref<tensor> > &OptimizationBase::parameters() {
    CXM_ASSERT(this->is_initialized == false, "Parameters isn't initialized");
    return this->m_params;
}

float32 OptimizationBase::lr() const {
    return this->m_lr;
}

const std::string &OptimizationBase::name() {
    return this->m_name;
}
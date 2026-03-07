//
// Created by muham on 25.02.2026.
//

#include "CortexMind/core/Net/optimization.hpp"
#include <utility>

using namespace cortex::_fw;

Optimization::Optimization(const float32 _lr, string info) : grads({}), params({}), lr(_lr), info(std::move(info)) {}

void Optimization::setParams(std::vector<ref<tensor> > _params, std::vector<ref<tensor> > _grads) {
    this->params = std::move(_params);
    this->grads  = std::move(_grads);
}

void Optimization::zero_grad() const {
    for (const auto &item : this->params) {
        if (item.get().has_grad()) item.get().zero_grad();
    }
}

cortex::string Optimization::config() const {
    return this->info;
}

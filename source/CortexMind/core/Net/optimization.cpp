//
// Created by muham on 18.03.2026.
//

#include "CortexMind/core/Net/optimization.hpp"
#include <type_traits>

using namespace cortex::_fw;
using namespace cortex;

Optimization::Optimization(const float32 _lr, std::string name) : lr(_lr), name(std::move(name)) {}

Optimization::Optimization(Optimization &&) noexcept = default;

Optimization::~Optimization() = default;

void Optimization::setParams(std::vector<ref<tensor>> _params, std::vector<ref<tensor>> _grads) {
    this->params = std::move(_params);
    this->grads  = std::move(_grads);
}

void Optimization::zero_grad() const {
    for (const auto& item : this->params) {
        if (item.get().grad_required()) {
            item.get().grad().zero();
        }
    }
}

const std::string &Optimization::config() const {
    return this->name;
}
//
// Created by muham on 25.05.2026.
//

#include "CortexMind/net/OptimizationFunction/momentum.hpp"
#include <string>

using namespace cortex::_fw;
using namespace cortex::opt;

Momentum::Momentum(const float32 lr, const float32 beta) : OptimizationBase("Momentum(" + std::to_string(lr) + ")", lr) {
    this->beta = beta;
    this->flag = flag;

    CXM_ASSERT(lr <= 0.0f, "Learning rate must be positive");
    CXM_ASSERT(beta < 0.0f || beta >= 1.0f, "Momentum beta must be in [0, 1)");
}

Momentum::~Momentum() = default;

void Momentum::update() {
    if (!this->flag) {
        this->Init();
    }

    for (size_t i = 0; i < this->m_params.size(); ++i) {
        tensor& param = this->m_params[i].get();
        tensor& v = this->velocities[i];

        if (param.grad().len() == 0) {
            continue;
        }

        v = this->beta * v + param.grad();

        param = param - this->m_lr * v;
    }
}

void Momentum::Init() {
    this->velocities.clear();

    for (const auto& item : this->m_params) {
        tensor v(item.get().shape(), item.get().device(), false);
        v.zero();
        this->velocities.push_back(v);
    }
    this->flag = true;
}

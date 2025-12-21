//
// Created by muham on 16.12.2025.
//

#include "CortexMind/net/OptimFunc/Adam/adam.hpp"
#include <CortexMind/framework/Tools/MathTools/math.hpp>
#include <CortexMind/framework/Tools/Debug/catch.hpp>
#include <iostream>
#include <cmath>

using namespace cortex::net;
using namespace cortex::_fw;
using namespace cortex;

Adam::Adam(const double lr, const double beta1, const double beta2, const double eps) : Optimizer(lr), b1(beta1), b2(beta2), epsilon(eps), t(0) {}

void Adam::zero_grad() {
    for (auto&[weights, gradients] : this->iters) {
        CXM_ASSERT(gradients != nullptr, "Gradient tensor is nullptr in Adam::zero_grad");
        gradients->zero();
    }
}

void Adam::step() {
    this->t++;

    const double b1t = std::pow(this->b1, this->t);
    const double b2t = std::pow(this->b2, this->t);

    for (size_t i = 0; i < this->iters.size(); ++i) {
        auto& weights = *this->iters[i].weights;
        auto& gradients = *this->iters[i].gradients;

        if (this->m_tensors.size() <= i) {
            this->m_tensors.emplace_back(weights.batch(), weights.channel(), weights.height(), weights.width());
            this->v_tensors.emplace_back(weights.batch(), weights.channel(), weights.height(), weights.width());
        }

        auto& m = this->m_tensors[i];
        auto& v = this->v_tensors[i];

        m = m * static_cast<float>(this->b1) + gradients * static_cast<float>(1.0 - this->b1);
        v = v * static_cast<float>(this->b2) + (gradients * gradients) * static_cast<float>(1.0 - this->b2);

        tensor m_hat = m / static_cast<float>(1 - b1t);
        tensor v_hat = v / static_cast<float>(1 - b2t);

        weights -= m_hat / (TensorFn::sqrt(v_hat) + static_cast<float>(this->epsilon)) * static_cast<float>(this->learning_rate);
    }
}

void Adam::add_param(tensor *weights, tensor *gradients) {
    CXM_ASSERT(weights != nullptr && gradients != nullptr, "Weights or gradients pointer is nullptr in Adam::add_param");
    this->iters.emplace_back(weights, gradients);
}

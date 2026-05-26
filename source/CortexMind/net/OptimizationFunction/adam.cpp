//
// Created by muham on 26.05.2026.
//

#include "CortexMind/net/OptimizationFunction/adam.hpp"
#include <cmath>
#include <string>

using namespace cortex::_fw;
using namespace cortex::opt;
using namespace cortex;

Adam::Adam(const float32 lr, const float32 beta1, const float32 beta2, const float32 eps) : OptimizationBase("Adam(" + std::to_string(lr) + ")", lr) {
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->eps = eps;
    this->t = 0;
    this->flag = false;
}

Adam::~Adam() = default;

void Adam::update() {
    if (!this->flag) {
        this->Init();
    }

    this->t++;

    const float32 bc1 = 1.0f - std::pow(this->beta1, static_cast<float32>(this->t));
    const float32 bc2 = 1.0f - std::pow(this->beta2, static_cast<float32>(this->t));
    const float32 alpha = this->m_lr * std::sqrt(bc2) / bc1;

    for (size_t i = 0; i < this->m_params.size(); ++i) {
        tensor& param = this->m_params[i].get();
        tensor& g = param.grad();

        this->m[i] = this->m[i] * this->beta1 + g * (1.0f - this->beta1);
        this->v[i] = this->v[i] * this->beta2 + (g * g) * (1.0f - this->beta2);

        param -= this->m[i] * alpha / (this->v[i].sqrt() + this->eps);
    }
}

void Adam::Init() {
    for (auto& item : this->m_params) {
        const tensor& x = item.get();
        this->m.emplace_back(x.shape(), x.device());
        this->m.back().zero();
        this->v.emplace_back(x.shape(), x.device());
        this->v.back().zero();
    }
    this->flag = true;
}

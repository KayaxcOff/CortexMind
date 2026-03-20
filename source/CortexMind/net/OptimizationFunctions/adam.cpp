//
// Created by muham on 18.03.2026.
//

#include "CortexMind/net/OptimizationFunctions/adam.hpp"
#include <cmath>

using namespace cortex::opt;
using namespace cortex;

Adam::Adam(const float32 lr, const float32 beta1, const float32 beta2, const float32 eps) : Optimization(lr, "Adam"), beta1(beta1), beta2(beta2), eps(eps), step(0) {}

Adam::~Adam() = default;
/*
void Adam::setParams(std::vector<_fw::ref<tensor>> _params, std::vector<_fw::ref<tensor>> _grads) {
    Optimization::setParams(std::move(_params), std::move(_grads));

    this->m_state.clear();
    this->v_state.clear();
    this->m_state.reserve(this->params.size());
    this->v_state.reserve(this->params.size());

    for (const auto& item : this->params) {
        tensor m(item.get().shape(), item.get().device());
        tensor v(item.get().shape(), item.get().device());
        m.zero();
        v.zero();
        this->m_state.push_back(std::move(m));
        this->v_state.push_back(std::move(v));
    }
}
*/
void Adam::update() {
    ++this->step;

    const float32 bc1 = 1.0f - std::pow(this->beta1, static_cast<float32>(this->step));
    const float32 bc2 = 1.0f - std::pow(this->beta2, static_cast<float32>(this->step));
    const float32 lr_t = this->lr * std::sqrt(bc2) / bc1;

    for (size_t i = 0; i < this->params.size(); ++i) {
        tensor& param = this->params[i].get();
        tensor& grad  = this->grads[i].get();

        this->m_state[i] = this->m_state[i] * this->beta1 + grad * (1.0f - this->beta1);

        this->v_state[i] = this->v_state[i] * this->beta2 + grad.pow(2.0f) * (1.0f - this->beta2);

        param -= this->m_state[i] * lr_t / (this->v_state[i].sqrt() + this->eps);
    }
}
//
// Created by muham on 5.03.2026.
//

#include "CortexMind/net/OptimizationFunctions/adam.hpp"
#include <string>

using namespace cortex::opt;
using namespace cortex::_fw;
using namespace cortex;

Adam::Adam(const float32 _lr, const float32 _beta1, const float32 _beta2, const float32 _eps) : Optimization(_lr, "Adam(" + std::to_string(_lr) + ")"), beta1(_beta1), beta2(_beta2), eps(_eps) {}

Adam::~Adam() = default;

void Adam::update() {
    this->Init();
    this->t++;

    const auto b1_corr = static_cast<float32>(1 - std::pow(this->beta1, this->t));
    const auto b2_corr = static_cast<float32>(1 - std::pow(this->beta2, this->t));

    for (size_t i = 0; i < this->params.size(); ++i) {
        auto& p = this->params[i].get();
        auto& g = this->grads[i].get();

        auto& mt = this->m[i];
        auto& vt = this->v[i];

        mt = this->beta1 * mt + (1.0f - this->beta1) * g;
        vt = this->beta2 * vt + (1.0f - this->beta2) * g.pow(2.0f);

        auto m_hat = mt / b1_corr;
        auto v_hat = vt / b2_corr;

        const auto update = this->lr * m_hat / (v_hat.sqrt() + this->eps);

        p -= update;
    }
}

void Adam::Init() {
    if (this->m.size() == this->params.size()) return;

    this->m.reserve(this->params.size());
    this->v.reserve(this->params.size());

    for (auto& item : this->params) {
        tensor mt(item.get().shape());
        tensor vt(item.get().shape());
        mt.zero();
        vt.zero();
        this->m.emplace_back(std::move(mt));
        this->v.emplace_back(std::move(vt));
    }
}

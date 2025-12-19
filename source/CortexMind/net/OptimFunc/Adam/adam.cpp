//
// Created by muham on 16.12.2025.
//

#include "CortexMind/net/OptimFunc/Adam/adam.hpp"
#include <CortexMind/framework/Core/AVX/funcs.hpp>
#include <CortexMind/framework/Tools/Debug/catch.hpp>
#include <iostream>
#include <cmath>

using namespace cortex::net;
using namespace cortex::_fw;
using namespace cortex;

Adam::Adam(const double lr, const double beta1, const double beta2, const double eps) : Optimizer(lr), b1(beta1), b2(beta2), epsilon(eps), t(0) {}

void Adam::zero_grad() {
    for (auto&[weights, gradients] : this->iters) {
        gradients->zero();
    }
}

void Adam::step() {
    this->t++;
    const auto b1t = static_cast<float>(std::pow(static_cast<float>(this->b1), this->t));
    const auto b2t = static_cast<float>(std::pow(static_cast<float>(this->b2), this->t));

    if (this->m_tensors.empty()) {
        for (auto&[weights, gradients] : iters) {
            this->m_tensors.emplace_back(weights->batch(), weights->channel(), weights->height(), weights->width());
            this->v_tensors.emplace_back(weights->batch(), weights->channel(), weights->height(), weights->width());
            this->m_tensors.back().zero();
            this->v_tensors.back().zero();
        }
    }

    for (size_t idx = 0; idx < this->iters.size(); idx++) {
        tensor* w = this->iters[idx].weights;
        tensor* g = this->iters[idx].gradients;
        tensor& m = this->m_tensors[idx];
        tensor& v = this->v_tensors[idx];

        if (!w || !g) {
            CXM_ASSERT(true, "w and g are nulll");
        }
        if (w->size() != g->size()) {
            CXM_ASSERT(true, "w and g size doesn't match");
        }
        if (w->size() != m.size() || w->size() != v.size()) {
            CXM_ASSERT(true, "w and m size doesn't match");
        }

        const size_t vec_size = w->vec_size();

        for (size_t i = 0; i < vec_size; i++) {
            const size_t rem = std::min(static_cast<size_t>(8), w->vec_size() - i*8);

            auto w_vec = (rem == 8) ? avx2::load(&w->dataIdx(i*8)[0]) : avx2::load_partial(&w->dataIdx(i*8)[0], rem);

            const auto g_vec = (rem == 8) ? avx2::load(&g->dataIdx(i*8)[0]) : avx2::load_partial(&g->dataIdx(i*8)[0], rem);
            auto m_vec = (rem == 8) ? avx2::load(&m.dataIdx(i*8)[0]) : avx2::load_partial(&m.dataIdx(i*8)[0], rem);
            auto v_vec = (rem == 8) ? avx2::load(&v.dataIdx(i*8)[0]) : avx2::load_partial(&v.dataIdx(i*8)[0], rem);

            m_vec = avx2::add(avx2::mul(avx2::broadcast(static_cast<float>(this->b1)), m_vec), avx2::mul(avx2::broadcast(1.0f - static_cast<float>(this->b1)), g_vec));

            const auto g_sq = avx2::mul(g_vec, g_vec);
            v_vec = avx2::add(avx2::mul(avx2::broadcast(static_cast<float>(this->b2)), v_vec), avx2::mul(avx2::broadcast(1.0f - static_cast<float>(this->b2)), g_sq));

            const auto m_hat = avx2::mul(m_vec, avx2::broadcast(1.0f / (1.0f - b1t)));
            const auto v_hat = avx2::mul(v_vec, avx2::broadcast(1.0f / (1.0f - b2t)));

            const auto denom = avx2::add(avx2::sqrt(v_hat), avx2::broadcast(static_cast<float>(this->epsilon)));
            const auto update = avx2::div(m_hat, denom);
            w_vec = avx2::sub(w_vec, avx2::mul(avx2::broadcast(static_cast<float>(this->learning_rate)), update));

            avx2::store(&m.dataIdx(i)[0], m_vec);
            avx2::store(&v.dataIdx(i)[0], v_vec);
            avx2::store(&w->dataIdx(i)[0], w_vec);
        }
    }
}

void Adam::add_param(tensor *weights, tensor *gradients) {
    this->iters.emplace_back(weights, gradients);
}

//
// Created by muham on 16.12.2025.
//

#include "CortexMind/net/OptimFunc/Momentum/momentum.hpp"
#include <CortexMind/framework/Core/AVX/funcs.hpp>
#include <CortexMind/framework/Tools/Debug/catch.hpp>
#include <iostream>

using namespace cortex::net;
using namespace cortex::_fw;
using namespace cortex;

Momentum::Momentum(const double lr, const double momentum) : Optimizer(lr), beta(momentum) {}

void Momentum::zero_grad() {
    for (auto&[weights, gradients] : this->iters) {
        gradients->zero();
    }
}

void Momentum::step() {
    CXM_ASSERT(this->iters.size() != this->velocities.size(), "Velocities not initialized! Call add_param first.");
    for (size_t idx = 0; idx < this->iters.size(); ++idx) {
        tensor* w = this->iters[idx].weights;
        tensor* g = this->iters[idx].gradients;
        tensor& v = this->velocities[idx];

        const size_t vec_size = w->vec_size();

        for (size_t i = 0; i < vec_size; ++i) {
            auto w_vec = avx2::load(&w->dataIdx(i)[0]);
            const auto g_vec = avx2::load(&g->dataIdx(i)[0]);
            auto v_vec = avx2::load(&v.dataIdx(i)[0]);

            v_vec = avx2::add(avx2::mul(avx2::broadcast(static_cast<float>(this->beta)), v_vec), avx2::mul(avx2::broadcast(1.0f - static_cast<float>(this->beta)), g_vec));

            w_vec = avx2::sub(w_vec, avx2::mul(avx2::broadcast(static_cast<float>(this->learning_rate)), v_vec));

            avx2::store(&v.dataIdx(i)[0], v_vec);
            avx2::store(&w->dataIdx(i)[0], w_vec);
        }
    }
}

void Momentum::add_param(tensor *weights, tensor *gradients) {
    this->iters.emplace_back(weights, gradients);
    this->velocities.emplace_back(weights->batch(), weights->channel(), weights->height(), weights->width(), 0.0f);
}

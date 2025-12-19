//
// Created by muham on 16.12.2025.
//

#include "CortexMind/net/OptimFunc/SGD/sgd.hpp"
#include <CortexMind/framework/Core/AVX/funcs.hpp>

using namespace cortex::net;
using namespace cortex::_fw;

void StaticGradient::step() {
    for (auto& [weight, grad] : this->iters) {
        for (size_t i = 0; i < weight->vec_size(); i++) {
            const auto w = avx2::load(&weight->dataIdx(i)[0]);
            const auto g = avx2::load(&grad->dataIdx(i)[0]);
            const auto updated = avx2::sub(w, avx2::mul(g, avx2::broadcast(static_cast<float>(learning_rate))));
            avx2::store(&grad->dataIdx(i)[0], updated);
        }
    }
}
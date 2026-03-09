//
// Created by muham on 8.03.2026.
//

#include "CortexMind/net/OptimizationFunctions/rms.hpp"
#include <string>

using namespace cortex::opt;
using namespace cortex::_fw;

RMSProp::RMSProp(const float32 _lr, const float32 _decay, const float32 eps) : Optimization(_lr, "RMSProp(" + std::to_string(_lr) + ")"), decay(_decay), eps(eps) {}

RMSProp::~RMSProp() = default;

void RMSProp::update() {
    if (this->cache.size() != this->params.size()) {
        this->cache.clear();
        for (const auto& item : this->params) {
            tensor c(item.get().shape(), false);
            c.zero();
            this->cache.emplace_back(std::move(c));
        }
    }

    for (size_t i = 0; i < this->params.size(); ++i) {
        tensor& w    = this->params[i].get();
        tensor& grad = this->grads[i].get();
        tensor& c    = this->cache[i];

        c *= this->decay;
        c += (grad * grad) * (1.0f - this->decay);

        tensor denom = c.sqrt() + this->eps;
        w -= (grad / denom) * this->lr;
    }
}

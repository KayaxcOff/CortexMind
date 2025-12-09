//
// Created by muham on 9.12.2025.
//

#ifndef CORTEXMIND_MSE_HPP
#define CORTEXMIND_MSE_HPP

#include <CortexMind/framework/Net/loss.hpp>

namespace cortex::net {
    class MeanSquared : public _fw::Loss {
    public:
        MeanSquared() = default;
        ~MeanSquared() override = default;

        tensor forward(const tensor &predictions, const tensor &targets) override;
        tensor backward(const tensor &predictions, const tensor &targets) override;
    };
}

#endif //CORTEXMIND_MSE_HPP
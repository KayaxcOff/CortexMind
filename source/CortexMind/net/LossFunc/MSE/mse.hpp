//
// Created by muham on 19.12.2025.
//

#ifndef CORTEXMIND_MSE_HPP
#define CORTEXMIND_MSE_HPP

#include <CortexMind/framework/Net/loss.hpp>

namespace cortex::net {
    class MeanSquared : public _fw::Loss {
    public:
        MeanSquared();
        ~MeanSquared() override;

        [[nodiscard]] tensor forward(const tensor &predictions, const tensor &targets) const override;
        [[nodiscard]] tensor backward(const tensor &predictions, const tensor &targets) const override;
    };
}

#endif //CORTEXMIND_MSE_HPP
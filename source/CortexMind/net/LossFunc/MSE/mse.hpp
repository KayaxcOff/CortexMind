//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_MSE_HPP
#define CORTEXMIND_MSE_HPP

#include <CortexMind/framework/NetBase/loss.hpp>

namespace cortex::loss {
    class MeanSquared : public Loss {
    public:
        MeanSquared() = default;
        ~MeanSquared() override = default;

        tensor forward(const tensor &predictions, const tensor &targets) override;
        tensor backward(const tensor &predictions, const tensor &targets) override;
    };
}

#endif //CORTEXMIND_MSE_HPP
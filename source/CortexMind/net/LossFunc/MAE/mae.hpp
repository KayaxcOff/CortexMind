//
// Created by muham on 19.12.2025.
//

#ifndef CORTEXMIND_MAE_HPP
#define CORTEXMIND_MAE_HPP

#include <CortexMind/framework/Net/loss.hpp>

namespace cortex::net {
    class MeanAbsolute : _fw::Loss {
    public:
        MeanAbsolute();
        ~MeanAbsolute() override;

        tensor backward(const tensor &predictions, const tensor &targets) const override;
        tensor forward(const tensor &predictions, const tensor &targets) const override;
    };
}

#endif //CORTEXMIND_MAE_HPP
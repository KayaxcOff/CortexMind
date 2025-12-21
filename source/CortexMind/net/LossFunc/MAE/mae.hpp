//
// Created by muham on 19.12.2025.
//

#ifndef CORTEXMIND_MAE_HPP
#define CORTEXMIND_MAE_HPP

#include <CortexMind/framework/Net/loss.hpp>

namespace cortex::net {
    class MeanAbsolute : public _fw::Loss {
    public:
        MeanAbsolute();
        ~MeanAbsolute() override;

        [[nodiscard]] tensor backward(const tensor &predictions, const tensor &targets) const override;
        [[nodiscard]] tensor forward(const tensor &predictions, const tensor &targets) const override;
    };
}

#endif //CORTEXMIND_MAE_HPP
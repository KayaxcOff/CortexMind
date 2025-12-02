//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_MAE_HPP
#define CORTEXMIND_MAE_HPP

#include <CortexMind/framework/NetBase/loss.hpp>

namespace cortex::loss {
    class MeanAbsolute : public Loss {
    public:
        MeanAbsolute() = default;
        ~MeanAbsolute() override = default;

        tensor forward(const tensor &predictions, const tensor &targets) override;
        tensor backward(const tensor &predictions, const tensor &targets) override;
    };
}

#endif //CORTEXMIND_MAE_HPP
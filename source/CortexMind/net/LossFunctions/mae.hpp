//
// Created by muham on 1.03.2026.
//

#ifndef CORTEXMIND_NET_LOSS_FUNCTIONS_MAE_HPP
#define CORTEXMIND_NET_LOSS_FUNCTIONS_MAE_HPP

#include <CortexMind/core/Net/loss.hpp>

namespace cortex::loss {
    class MeanAbsolute : public _fw::Loss {
    public:
        MeanAbsolute();
        ~MeanAbsolute() override;

        tensor forward(const tensor &predicted, const tensor &target) override;
    };
} // namespace cortex::loss

#endif //CORTEXMIND_NET_LOSS_FUNCTIONS_MAE_HPP
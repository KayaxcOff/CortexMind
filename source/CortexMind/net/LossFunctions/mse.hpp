//
// Created by muham on 19.03.2026.
//

#ifndef CORTEXMIND_NET_LOSS_FUNCTIONS_MSE_HPP
#define CORTEXMIND_NET_LOSS_FUNCTIONS_MSE_HPP

#include <CortexMind/core/Net/loss.hpp>

namespace cortex::loss {
    class MeanSquared : public _fw::Loss {
    public:
        MeanSquared();
        ~MeanSquared() override;

        [[nodiscard]]
        tensor forward(tensor &predicted, tensor &target) override;
    };
} // namespace cortex::loss

#endif //CORTEXMIND_NET_LOSS_FUNCTIONS_MSE_HPP
//
// Created by muham on 1.03.2026.
//

#ifndef CORTEXMIND_NET_LOSS_FUNCTIONS_MSE_HPP
#define CORTEXMIND_NET_LOSS_FUNCTIONS_MSE_HPP

#include <CortexMind/core/Net/loss.hpp>

namespace cortex::loss {
    class MeanSquared : public _fw::Loss {
    public:
        MeanSquared();
        ~MeanSquared() override;

        tensor forward(const tensor &predicted, const tensor &target) override;
    private:
        tensor last_diff;
        tensor last_sq;
    };
} // namespace cortex::loss

#endif //CORTEXMIND_NET_LOSS_FUNCTIONS_MSE_HPP
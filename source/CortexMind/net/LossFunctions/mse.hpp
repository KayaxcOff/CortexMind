//
// Created by muham on 5.05.2026.
//

#ifndef CORTEXMIND_NET_LOSS_FUNCTIONS_MSE_HPP
#define CORTEXMIND_NET_LOSS_FUNCTIONS_MSE_HPP

#include <CortexMind/framework/Net/loss.hpp>

namespace cortex::loss {
    class MeanSquared : public _fw::LossBase {
    public:
        MeanSquared();
        ~MeanSquared() override;

        tensor forward(tensor &prediction, tensor &target) override;
    };
} //namespace cortex::loss

#endif //CORTEXMIND_NET_LOSS_FUNCTIONS_MSE_HPP
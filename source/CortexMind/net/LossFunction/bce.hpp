//
// Created by muham on 25.05.2026.
//

#ifndef CORTEXMIND_NET_LOSS_FUNCTION_BCE_HPP
#define CORTEXMIND_NET_LOSS_FUNCTION_BCE_HPP

#include <CortexMind/framework/Net/loss.hpp>

namespace cortex::loss {
    class BinaryCrossEntropy : public _fw::LossBase {
    public:
        explicit BinaryCrossEntropy(float32 eps = 1e-7f);
        ~BinaryCrossEntropy() override;

        [[nodiscard]]
        tensor forward(const tensor &predict, const tensor &target) override;
    private:
        float32 eps;
    };
} //namespace cortex::loss

#endif //CORTEXMIND_NET_LOSS_FUNCTION_BCE_HPP
//
// Created by muham on 6.03.2026.
//

#ifndef CORTEXMIND_NET_LOSS_FUNCTIONS_BCE_HPP
#define CORTEXMIND_NET_LOSS_FUNCTIONS_BCE_HPP

#include <CortexMind/core/Net/loss.hpp>

namespace cortex::loss {
    class BinaryCrossEntropy : public _fw::Loss {
    public:
        BinaryCrossEntropy();
        ~BinaryCrossEntropy() override;

        tensor forward(const tensor &predicted, const tensor &target) override;
    };
} // namespace cortex::loss

#endif //CORTEXMIND_NET_LOSS_FUNCTIONS_BCE_HPP
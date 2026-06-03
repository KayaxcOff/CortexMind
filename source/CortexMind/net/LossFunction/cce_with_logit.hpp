//
// Created by muham on 3.06.2026.
//

#ifndef CORTEXMIND_NET_LOSS_FUNCTION_CCE_WITH_LOGIT_HPP
#define CORTEXMIND_NET_LOSS_FUNCTION_CCE_WITH_LOGIT_HPP

#include <CortexMind/framework/Net/loss.hpp>

namespace cortex::loss {
    class CategoricalCrossEntropyWithLogit : public _fw::LossBase {
    public:
        CategoricalCrossEntropyWithLogit();
        ~CategoricalCrossEntropyWithLogit() override;

        [[nodiscard]]
        tensor forward(const tensor &predict, const tensor &target) override;
    };
} //namespace cortex::loss

#endif //CORTEXMIND_NET_LOSS_FUNCTION_CCE_WITH_LOGIT_HPP
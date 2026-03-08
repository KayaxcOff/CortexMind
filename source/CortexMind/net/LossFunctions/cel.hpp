//
// Created by muham on 8.03.2026.
//

#ifndef CORTEXMIND_NET_LOSS_FUNCTIONS_CEL_HPP
#define CORTEXMIND_NET_LOSS_FUNCTIONS_CEL_HPP

#include <CortexMind/core/Net/loss.hpp>

namespace cortex::loss {
    class CrossBinary : public _fw::Loss {
    public:
        CrossBinary();
        ~CrossBinary() override;

        tensor forward(const tensor &predicted, const tensor &target) override;
    };
} // namespace cortex::loss

#endif //CORTEXMIND_NET_LOSS_FUNCTIONS_CEL_HPP
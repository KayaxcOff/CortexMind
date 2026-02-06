//
// Created by muham on 6.02.2026.
//

#ifndef CORTEXMIND_CORE_NET_LOSS_HPP
#define CORTEXMIND_CORE_NET_LOSS_HPP

#include <CortexMind/tools/params.hpp>

namespace cortex::_fw {
    class Loss {
    public:
        Loss() = default;

        virtual
        ~Loss() = default;

        [[nodiscard]] virtual
        tensor forward(const tensor& predictions, const tensor& targets) = 0;

        [[nodiscard]] virtual
        tensor backward(const tensor& predictions, const tensor& targets) = 0;
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_NET_LOSS_HPP
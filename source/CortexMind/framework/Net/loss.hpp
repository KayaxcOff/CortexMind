//
// Created by muham on 10.12.2025.
//

#ifndef CORTEXMIND_LOSS_HPP
#define CORTEXMIND_LOSS_HPP

#include <CortexMind/framework/Params/params.hpp>

namespace cortex::_fw {
    class Loss {
    public:
        Loss() = default;
        virtual ~Loss() = default;

        [[nodiscard]] virtual tensor forward(const tensor& predictions, const tensor& targets) const = 0;
        [[nodiscard]] virtual tensor backward(const tensor& predictions, const tensor& targets) const = 0;
    };
}

#endif //CORTEXMIND_LOSS_HPP
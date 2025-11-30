//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_LOSS_HPP
#define CORTEXMIND_LOSS_HPP

#include <CortexMind/framework/Params/params.hpp>

namespace cortex::loss {
    class Loss {
    public:
        Loss() = default;
        virtual ~Loss() = default;

        virtual tensor forward(const tensor& predictions, const tensor& targets) = 0;
        virtual tensor backward(const tensor& predictions, const tensor& targets) = 0;
    };
}

#endif //CORTEXMIND_LOSS_HPP
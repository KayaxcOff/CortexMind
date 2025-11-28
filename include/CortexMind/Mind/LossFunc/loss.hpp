//
// Created by muham on 8.11.2025.
//

#ifndef CORTEXMIND_LOSS_HPP
#define CORTEXMIND_LOSS_HPP

#include <CortexMind/Utils/params.hpp>

namespace cortex::loss {
    class Loss {
    public:
        Loss();
        virtual ~Loss();

        virtual tensor forward(const tensor& y_true, const tensor& y_pred) = 0;
        virtual tensor backward(const tensor& y_true, const tensor& y_pred) = 0;
    };
}

#endif //CORTEXMIND_LOSS_HPP
//
// Created by muham on 3.11.2025.
//

#ifndef CORTEXMIND_LOSS_HPP
#define CORTEXMIND_LOSS_HPP

#include "../../Utils/MathTools/vector/vector.hpp"

namespace cortex::loss {
    class Loss {
    public:
        virtual ~Loss() = default;

        virtual double forward(const math::MindVector &predictions, const math::MindVector &targets) = 0;
        virtual math::MindVector backward(const math::MindVector &predictions, const math::MindVector &targets) = 0;
    };
}

#endif //CORTEXMIND_LOSS_HPP
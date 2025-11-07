//
// Created by muham on 3.11.2025.
//

#ifndef CORTEXMIND_MSE_HPP
#define CORTEXMIND_MSE_HPP

#include "../loss.hpp"
#include "../../../Utils/MathTools/vector/vector.hpp"

namespace cortex::loss {
    class MeanSquared final : public Loss {
    public:
        MeanSquared();

        double forward(const math::MindVector &predictions, const math::MindVector &targets) override;
        math::MindVector backward(const math::MindVector &predictions, const math::MindVector &targets) override;
    };
}

#endif //CORTEXMIND_MSE_HPP
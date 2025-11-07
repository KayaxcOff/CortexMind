//
// Created by muham on 4.11.2025.
//

#ifndef CORTEXMIND_MAE_HPP
#define CORTEXMIND_MAE_HPP

#include "../loss.hpp"

namespace cortex::loss {
    class MeanAbsolute final : public Loss {
    public:
        double forward(const math::MindVector &predictions, const math::MindVector &targets) override;
        math::MindVector backward(const math::MindVector &predictions, const math::MindVector &targets) override;
    };
}

#endif //CORTEXMIND_MAE_HPP
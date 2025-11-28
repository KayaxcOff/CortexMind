//
// Created by muham on 28.11.2025.
//

#ifndef CORTEXMIND_MSE_HPP
#define CORTEXMIND_MSE_HPP

#include "loss.hpp"

namespace cortex::loss {
    class MeanSquared final : public Loss {
    public:
        MeanSquared();
        ~MeanSquared() override;

        tensor forward(const tensor &y_true, const tensor &y_pred) override;
        tensor backward(const tensor &y_true, const tensor &y_pred) override;
    };
}

#endif //CORTEXMIND_MSE_HPP
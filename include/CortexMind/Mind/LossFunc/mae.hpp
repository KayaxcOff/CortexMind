//
// Created by muham on 28.11.2025.
//

#ifndef CORTEXMIND_MAE_HPP
#define CORTEXMIND_MAE_HPP

#include "loss.hpp"

namespace cortex::loss {
    class MeanAbsolute : public Loss {
    public:
        MeanAbsolute();
        ~MeanAbsolute() override;

        tensor forward(const tensor &y_true, const tensor &y_pred) override;
        tensor backward(const tensor &y_true, const tensor &y_pred) override;
    };
}

#endif //CORTEXMIND_MAE_HPP